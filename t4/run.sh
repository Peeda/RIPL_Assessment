#!/usr/bin/env bash
# ==============================================================================
# T-IV - Policy Decorator residual, PPO, on T-III's reward and sampler.
#
#   bash t4/run.sh test        the offline suite. No sim, ~5 s. Needs torch,
#                              which the pod venv has; on the laptop wrap it in
#                              nix-shell (see CLAUDE.md)
#   bash t4/run.sh capture     physx_cpu initial states of the nominal blocks
#   bash t4/run.sh backend     replay them on physx_cuda. THE GATE - run it
#                              BEFORE spending GPU hours
#   bash t4/run.sh smoke       one tiny PPO iteration end to end
#   bash t4/run.sh train       3 residual seeds for $MODE
#   bash t4/run.sh eval        the paired T-II evaluation, physx_cpu
#   bash t4/run.sh verify      t2/verify.py on the after pass, then verify_t4.py
#   bash t4/run.sh report      the figures and the before/after table
#   bash t4/run.sh all         capture, backend, smoke, train, eval, verify, report
#
# TRAINING IS GPU, SCORING IS CPU, and `backend` is what licenses that. A seed
# does not address an episode on physx_cuda and the assertion that should catch
# it passes and lies, so the seed-addressed harness cannot be ported - see
# t4/backend_check.py.
#
# Two traps, both from CLAUDE.md, both handled here:
#   * the after pass writes to its OWN $T2_OUT. eval_modes.py refuses to
#     overwrite a finished block, so pointing it at t2/results would silently
#     skip everything and reprint the BASE numbers.
#   * seeds.csv is COPIED, never rebuilt. Selection is deterministic given the
#     index; an index rebuilt at a different size selects different seeds and
#     the comparison stops being paired.
# ==============================================================================
set -euo pipefail

ENVSH="${RIPL_ROOT:-/workspace/ripl}/env.sh"
# shellcheck source=/dev/null
[ -f "$ENVSH" ] && source "$ENVSH"

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
SELF=$HERE/$(basename "$0")

MODE=${MODE:-gap}
GEN=${GEN:-2}
export T3_RUN=${T3_RUN:-$ROOT/t3/artifacts/${MODE}${GEN:+_gen$GEN}}
export T3_SAMPLER=${T3_SAMPLER:-1}
export T4_NOMINAL_FRAC=${T4_NOMINAL_FRAC:-0}

RUNS=${T4_RUNS:-$ROOT/t4/runs}
RESULTS=${T4_RESULTS:-$ROOT/t4/results}
T2_SRC=${T2_SRC:-$ROOT/t2/results}
AFTER=${T4_AFTER:-$RESULTS/after_$MODE}

SEEDS=${SEEDS:-1 2 3}
ALPHA=${ALPHA:-0.05}
RES_HORIZON=${RES_HORIZON:-0}
NUM_ENVS=${NUM_ENVS:-64}
TOTAL_STEPS=${TOTAL_STEPS:-4000000}
ALPHA_WARMUP=${ALPHA_WARMUP:-1000000}
TRACK=${TRACK:-}
export STATE_FLAG=${STATE_FLAG:-}

t0=$(date +%s)
stage() { STAGE="$1"; echo ""; echo "######## $* [$(( $(date +%s) - t0 ))s]"; }

on_exit() {
  local rc=$?
  [ "$rc" -eq 0 ] && return 0
  echo ""
  echo "######## FAILED at stage '${STAGE:-?}' (exit $rc) after $(( $(date +%s) - t0 ))s"
  echo ""
  echo "  Trained residuals live in $RUNS and evaluation blocks in $AFTER."
  echo "  Neither is reproducible - PPO and DDPM are both stochastic. Pull"
  echo "  before you stop or terminate the pod:"
  echo "      bash setup/transfer.sh info"
  echo ""
  echo "  Re-running resumes: finished evaluation blocks are skipped."
  exit "$rc"
}
trap on_exit EXIT

# --- the frozen base -------------------------------------------------------
REPO_CKPT=$ROOT/checkpoints/stackcube_rgb_spatial_800demos.pt
find_ckpt() {
  if [ -n "${CKPT:-}" ]; then :
  elif [ -f "$REPO_CKPT" ]; then export CKPT="$REPO_CKPT"
  else
    echo "!! no base checkpoint. Set CKPT, or restore $REPO_CKPT." >&2
    exit 1
  fi
  echo "  base policy   $CKPT"
}

do_preflight() {
  stage "preflight"
  : "${MANISKILL_REPO:?set MANISKILL_REPO (source \$RIPL_ROOT/env.sh)}"
  find_ckpt
  [ -f "$T3_RUN/reward.py" ] || { echo "!! no reward.py in T3_RUN=$T3_RUN" >&2; exit 1; }
  [ -f "$T3_RUN/sampler.py" ] || { echo "!! no sampler.py in T3_RUN=$T3_RUN" >&2; exit 1; }
  python3 "$ROOT/t3/loader.py" "$T3_RUN" >/dev/null
  # patches/0001 is what makes the spatial encoder loadable; setup_runpod.sh
  # re-clones ManiSkill and wipes it, so check rather than discover at
  # load_state_dict time.
  python - <<'PY'
import dataclasses, os, sys
sys.path.insert(0, f"{os.environ['MANISKILL_REPO']}/examples/baselines/diffusion_policy")
import train_rgbd as T
if not any(f.name == "pool_feature_map" for f in dataclasses.fields(T.Args)):
    sys.exit("\n!! patches/0001 is not applied to $MANISKILL_REPO, so the "
             "spatial-encoder\n   checkpoint will not load. Run "
             "'bash setup/apply_patches.sh' first.\n")
PY
  echo "  T3_RUN        $T3_RUN"
  echo "  T3_SAMPLER    $T3_SAMPLER   T4_NOMINAL_FRAC $T4_NOMINAL_FRAC"
  mkdir -p "$RUNS/$MODE" "$RESULTS"
}

do_test() {
  stage "test - the offline suite"
  python3 "$HERE/test_t4.py"
  python3 "$ROOT/t2/test_geometry.py" | tail -2
}

do_capture() {
  stage "capture - physx_cpu initial states of the nominal blocks"
  local npz="$RESULTS/nominal_states.npz"
  if [ -f "$npz" ]; then echo "  skip - $npz exists"; return; fi
  BACKEND=physx_cpu python "$HERE/capture_states.py" \
    --csv "$T2_SRC"/mode_nominal_seed1.csv "$T2_SRC"/mode_nominal_seed2.csv \
          "$T2_SRC"/mode_nominal_seed3.csv \
    --out "$npz" --num-envs "${CAPTURE_ENVS:-10}"
}

do_backend() {
  stage "backend - replay those states on physx_cuda. THE GATE"
  find_ckpt
  BACKEND=physx_cuda python "$HERE/backend_check.py" "$CKPT" $STATE_FLAG \
    --states "$RESULTS/nominal_states.npz" \
    --cpu-csv "$T2_SRC"/mode_nominal_seed1.csv "$T2_SRC"/mode_nominal_seed2.csv \
              "$T2_SRC"/mode_nominal_seed3.csv \
    --num-envs "${BACKEND_ENVS:-64}" --limit "${BACKEND_LIMIT:-0}" \
    --out "$RESULTS/backend_check"
  echo ""
  echo "  Read the agreement against the ~0.74 same-backend floor, not against"
  echo "  1.0, and read the CONDITIONAL table - the marginal agreeing while the"
  echo "  failure region moves is the outcome that invalidates T-IV."
}

do_smoke() {
  stage "smoke - one tiny PPO iteration, end to end"
  find_ckpt
  rm -rf "$RUNS/smoke"
  python "$HERE/train_ppo.py" --mode "$MODE" --seed 99 --ckpt "$CKPT" \
    --out "$RUNS/smoke" --num-envs 4 --total-timesteps 6400 \
    --alpha "$ALPHA" --alpha-warmup 3200 --num-minibatches 2 --update-epochs 1
  echo "  smoke output is in $RUNS/smoke and is NOT a result; delete it freely."
}

do_train() {
  find_ckpt
  for s in $SEEDS; do
    stage "train - mode '$MODE' residual seed $s"
    if [ -f "$RUNS/$MODE/residual_seed$s.pt" ] && [ "${FORCE:-}" != "1" ]; then
      echo "  skip - residual_seed$s.pt exists. FORCE=1 to retrain."
      continue
    fi
    python "$HERE/train_ppo.py" --mode "$MODE" --seed "$s" --ckpt "$CKPT" \
      --out "$RUNS/$MODE" --num-envs "$NUM_ENVS" \
      --total-timesteps "$TOTAL_STEPS" --alpha "$ALPHA" \
      --alpha-warmup "$ALPHA_WARMUP" --res-horizon "$RES_HORIZON" ${TRACK:+--track}
  done
}

do_eval() {
  stage "eval - the paired T-II evaluation of mode '$MODE', physx_cpu"
  find_ckpt
  mkdir -p "$AFTER"
  # COPIED, not rebuilt: selection is deterministic given the index, and an
  # index rebuilt at a different size selects different seeds.
  [ -f "$AFTER/seeds.csv" ] || cp "$T2_SRC/seeds.csv" "$AFTER/seeds.csv"
  for s in $SEEDS; do
    [ -f "$RUNS/$MODE/residual_seed$s.pt" ] || {
      echo "!! $RUNS/$MODE/residual_seed$s.pt is missing; train first." >&2; exit 1; }
  done
  # A SEPARATE T2_OUT. Pointing this at t2/results would skip every finished
  # block and reprint the base numbers - silently.
  T2_OUT="$AFTER" \
  RESIDUAL="$RUNS/$MODE/residual_seed{block}.pt" \
  BACKEND=physx_cpu \
    python "$ROOT/t2/eval_modes.py" "$CKPT" $STATE_FLAG \
      --modes ${MODES:-nominal gap farb} --index "$AFTER/seeds.csv" \
      --out "$AFTER" --episodes "${EPISODES:-100}" --blocks "${BLOCKS:-3}" \
      --num-envs "${EVAL_ENVS:-10}"
}

do_verify() {
  stage "verify"
  python3 "$ROOT/t2/verify.py" "$AFTER"
  python3 "$HERE/verify_t4.py" "$T2_SRC" "$AFTER"
}

do_report() {
  stage "report"
  local args=()
  for d in "$RESULTS"/after_*; do
    [ -d "$d" ] && args+=("$(basename "$d" | sed 's/^after_//')=$d")
  done
  python3 "$HERE/report.py" --runs "$RUNS" --before "$T2_SRC" \
    --figdir "$ROOT/figures" ${args[@]:+--after "${args[@]}"}
}

case "${1:-all}" in
  test)      do_test ;;
  capture)   do_preflight; do_capture ;;
  backend)   do_preflight; do_capture; do_backend ;;
  smoke)     do_preflight; do_smoke ;;
  train)     do_preflight; do_train ;;
  eval)      do_preflight; do_eval ;;
  verify)    do_verify ;;
  report)    do_report ;;
  all)       do_test; do_preflight; do_capture; do_backend; do_smoke
             do_train; do_eval; do_verify; do_report ;;
  *) trap - EXIT; sed -n '2,27p' "$SELF"; exit 1 ;;
esac

echo ""
echo "######## done in $(( $(date +%s) - t0 ))s"
echo "  Figures and run logs do NOT come back through git. Before stopping the"
echo "  pod:  bash setup/transfer.sh info"
