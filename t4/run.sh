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
#   bash t4/run.sh train       the residual(s) for $MODE ($SEEDS, default 1)
#   bash t4/run.sh eval        the paired T-II evaluation, physx_cpu
#   bash t4/run.sh verify      t2/verify.py on the after pass, then verify_t4.py
#   bash t4/run.sh report      the figures and the before/after table
#   bash t4/run.sh all         capture, backend, smoke, train, eval, verify, report
#
# TRAINING AND SCORING ARE BOTH physx_cpu. `backend` is what decided that: it
# measured the frozen base at 0.730 on cpu against 0.557 on cuda over the SAME
# 300 initial states, with the loss concentrated after placement
# (success|placed 0.849 -> 0.711) and the grasp rate HOLDING - contact physics,
# not perception, and not a settable config difference. That gap lands exactly
# on the stage T-II's `farb` mode is defined by, so a residual trained on GPU
# would be learning to fix a backend artifact. Run `backend` before trusting
# any of this on a new pod.
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

# ONE training seed per mode. backend_check measured a 17-point cpu/gpu gap
# concentrated after placement, so training is physx_cpu and ~40x slower; the
# budget bought one residual per mode rather than three. The three evaluation
# blocks therefore vary only the initial states and the DDPM noise, and the
# report must say that the SD excludes TRAINING variance.
SEEDS=${SEEDS:-1}
ALPHA=${ALPHA:-0.05}
# The EXPLORATION scale, and the thing the alpha ramp does NOT bound. The head
# samples raw ~ N(mu, exp(LOG_STD)) per axis PER ENV STEP, so at the -1.0 that
# upstream ppo.py uses the per-step perturbation is 0.368 * alpha = 1.84 mm and
# the random walk over a 200-step episode is 1.84*sqrt(200) = 26 mm - larger
# than the <20 mm face clearance that DEFINES mode `gap`. Exploration noise
# scaling with alpha is why the alpha ramp alone did not protect the base
# policy; see t4/README.md section 3.
LOG_STD=${LOG_STD:--2.5}
# Upstream ppo.py's 3e-4. Raised here only with evidence: the first 1M-step run
# left the head at its near-zero init (approx_kl ~1e-4, and a paired eval of the
# result moved gap block 1 by -0.010), so the mean was not moving at all. The
# head is ~420k params with a critic already at EV 0.6-0.8, and target_kl 0.1
# catches an overshoot.
LR=${LR:-3e-4}
RES_HORIZON=${RES_HORIZON:-0}
# physx_cpu vectorises by SUBPROCESS, so this is a process count, not a batch
# width. It buys nothing above the core count.
NUM_ENVS=${NUM_ENVS:-16}
# `smoke` is separate from $NUM_ENVS so the end-to-end check stays cheap, but it
# is a knob because the ONLY honest way to size $TOTAL_STEPS is to measure the
# rate at the width you will train at.
SMOKE_ENVS=${SMOKE_ENVS:-4}
MAX_EP_STEPS=${MAX_EP_STEPS:-200}
TRAIN_BACKEND=${TRAIN_BACKEND:-physx_cpu}
TOTAL_STEPS=${TOTAL_STEPS:-1000000}
ALPHA_WARMUP=${ALPHA_WARMUP:-250000}
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
  python3 "$HERE/test_train_loop.py"
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

# physx_cpu vectorises by SUBPROCESS, and every worker's numpy/torch starts an
# OpenBLAS and an OpenMP pool sized to nproc. At 64 workers on a 128-core box
# that is ~8,000 threads, and the container's cgroup pids limit refuses them:
#
#   OpenBLAS blas_thread_init: pthread_create failed ... Resource temporarily
#   unavailable ... RLIMIT_NPROC -1 current, -1 max
#
# RLIMIT_NPROC being -1 (unlimited) is the tell that the ceiling is the cgroup,
# which cannot be raised from inside the container. So cap the pools instead.
# A worker steps ONE env, so its BLAS has nothing to parallelise and the pool
# was pure oversubscription - expect the step rate to RISE, not fall.
#
# Scoped to training on purpose. The T-II before-pass was measured without
# these, and BLAS thread count changes reduction order, hence the last bits of
# a float, hence the DDPM trajectory. Setting them for `eval` would leave the
# after-pass on a fractionally different numerical path from the committed
# before-pass it is compared against.
limit_threads() {
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
         NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
  echo "  threads       BLAS/OMP pools capped at 1 per worker"
}

# The OTHER per-worker resource, and the one that is invisible where you would
# look for it. Every physx_cpu worker builds its own SAPIEN Vulkan renderer
# with its own GPU buffers, so VRAM also scales with the PROCESS count:
#
#   CUDA error at .../sapien-vulkan-2/src/core/buffer.cpp 251: out of memory
#
# torch.cuda.max_memory_allocated() - what train_ppo logs as sys/vram_max_gb -
# does NOT see any of it; it reports the residual head and the DDPM only. So a
# run can read 0.1 GB right up until the renderer OOMs. Read the ceiling here
# instead, and quote nvidia-smi rather than sys/vram_max_gb in the report.
gpu_note() {
  command -v nvidia-smi >/dev/null || return 0
  nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free \
             --format=csv,noheader | while IFS= read -r line; do
    echo "  gpu           $line"
  done
  echo "  NOTE  each worker builds its own Vulkan renderer, so VRAM scales with"
  echo "        \$NUM_ENVS. If this OOMs in buffer.cpp, lower the width - watch"
  echo "        nvidia-smi during a small run to get the per-worker cost."
}

do_smoke() {
  stage "smoke - three tiny PPO iterations, end to end, at \$SMOKE_ENVS=$SMOKE_ENVS"
  find_ckpt
  limit_threads
  gpu_note
  rm -rf "$RUNS/smoke"
  # One iteration is one episode per env, so it is $MAX_EP_STEPS env steps per
  # env WHATEVER $RES_HORIZON is. Three of them, so the printed rate is not
  # dominated by the first iteration's warm-up.
  local total=$(( SMOKE_ENVS * MAX_EP_STEPS * 3 ))
  python "$HERE/train_ppo.py" --mode "$MODE" --seed 99 --ckpt "$CKPT" \
    --out "$RUNS/smoke" --num-envs "$SMOKE_ENVS" --total-timesteps "$total" \
    --sim-backend "$TRAIN_BACKEND" --res-horizon "$RES_HORIZON" \
    --alpha "$ALPHA" --alpha-warmup $(( total / 2 )) \
    --num-minibatches 2 --update-epochs 1
  echo ""
  echo "  env-step/s above is the AGGREGATE over all $SMOKE_ENVS processes -"
  echo "  global_step counts num_envs per sub-step - so do NOT multiply it by"
  echo "  the process count again. Divide \$TOTAL_STEPS by it to size the run,"
  echo "  and re-measure at the \$NUM_ENVS you actually intend to train at:"
  echo "  physx_cpu scales by subprocess and the rate is sublinear in it."
  echo "  smoke output is in $RUNS/smoke and is NOT a result; delete it freely."
}

do_train() {
  find_ckpt
  limit_threads
  gpu_note
  for s in $SEEDS; do
    stage "train - mode '$MODE' residual seed $s"
    if [ -f "$RUNS/$MODE/residual_seed$s.pt" ] && [ "${FORCE:-}" != "1" ]; then
      echo "  skip - residual_seed$s.pt exists. FORCE=1 to retrain."
      continue
    fi
    python "$HERE/train_ppo.py" --mode "$MODE" --seed "$s" --ckpt "$CKPT" \
      --out "$RUNS/$MODE" --num-envs "$NUM_ENVS" \
      --sim-backend "$TRAIN_BACKEND" \
      --total-timesteps "$TOTAL_STEPS" --alpha "$ALPHA" \
      --alpha-warmup "$ALPHA_WARMUP" --res-horizon "$RES_HORIZON" \
      --log-std-init "$LOG_STD" --learning-rate "$LR" ${TRACK:+--track}
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
  # With ONE training seed the same head runs every block; with three, seed b
  # is paired to block b so the spread carries training variance too. Chosen
  # from $SEEDS rather than hardcoded, because getting it wrong silently
  # evaluates the wrong head.
  local nseeds
  nseeds=$(echo $SEEDS | wc -w)
  if [ "$nseeds" -eq 1 ]; then
    RES_PATTERN="$RUNS/$MODE/residual_seed$SEEDS.pt"
  else
    RES_PATTERN="$RUNS/$MODE/residual_seed{block}.pt"
  fi
  echo "  residual      $RES_PATTERN"
  # A SEPARATE T2_OUT. Pointing this at t2/results would skip every finished
  # block and reprint the base numbers - silently.
  T2_OUT="$AFTER" \
  RESIDUAL="$RES_PATTERN" \
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
