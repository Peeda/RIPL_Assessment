#!/usr/bin/env bash
# ==============================================================================
# replay_pusht.sh - full-scale PushT-v1 demo replay for T-I.
#
# Takes the shipped obs_mode=none demonstrations (actions + recorded sim states,
# no observations) and replays them to attach observations, producing the two
# datasets T-I trains on: one state, one rgb. Downloading is incidental; the
# replay is the work, which is why this is no longer called fetch_*.
#
# Run AFTER smoke_test_e2e.sh is green:
#   source /workspace/ripl/env.sh && tmux new -s replay && bash replay_pusht.sh
#
# The RGB pass is the long pole of day 1. Start it and go do something else.
# ==============================================================================
set -uo pipefail   # not -e: the state pass should still count if RGB dies

: "${MS_ASSET_DIR:?source /workspace/ripl/env.sh first}"
DEMOS=$MS_ASSET_DIR/demos/PushT-v1/rl
BACKEND=physx_cuda
STATE_ENVS=${STATE_ENVS:-1024}
RGB_ENVS=${RGB_ENVS:-256}

# -----------------------------------------------------------------------------
# CONTROL MODE. Pick once; use it unchanged from T-I through T-IV.
#
#   pd_ee_delta_pos    3-DoF, translation only
#   pd_ee_delta_pose   6-DoF, translation + rotation
#
# Default here is pd_ee_delta_pos, for two reasons that point the same way:
#
#   1. T-I's deliverable is a VISUAL policy. The maintainers use
#      pd_ee_delta_pos for their RGB PushT baselines (pd_ee_delta_pose for
#      state), so this keeps the number comparable to what they publish.
#   2. T-IV's residual is a = a_base + clip(D, -alpha, alpha), and alpha is a
#      physical bound in action space. Under 3-DoF it bounds translation, full
#      stop. Under 6-DoF one scalar bounds translation AND rotation, which have
#      no shared scale, and you would need a per-axis bound to say anything
#      coherent about it. PushT is planar; the rotation DoFs buy nothing.
#
# If you switch back to pose, switch it in smoke_test_e2e.sh too - a mismatch
# between replay and eval is exactly what gate 9 exists to catch.
# -----------------------------------------------------------------------------
CTRL=${CTRL:-pd_ee_delta_pos}

# --use-env-states (not --use-first-env-state) is the flag that matters. It
# resets the simulator to the recorded state at EVERY timestep, so numerical
# divergence cannot accumulate through contact events. With --use-first-env-
# state the replay runs open-loop from t=0 and ~60% of PushT demos drift off
# course and get discarded; with --use-env-states yield is ~97%.
# These are the flags from scripts/data_generation/replay_for_il_baselines.sh,
# which is what generated the maintainers' published baseline datasets.
#
# --max-retry is mostly redundant once --use-env-states is set, but harmless.
MAX_RETRY=${MAX_RETRY:-5}

# A replay output existing is not the same as it being complete. --use-env-
# states yields 96-98%, so anything much under 85% of the source episode count
# came from a capped run or the old open-loop flag, and reusing it silently
# caps your T-I baseline.
#
# The threshold is DERIVED, not hardcoded: PushT ships a different number of
# episodes per control mode (888 for pd_ee_delta_pos, 719 for pd_ee_delta_pose,
# 999 for pd_joint_delta_pos), so any fixed number is wrong for at least two of
# them. Set MIN_TRAJS explicitly to override.
MIN_TRAJS=${MIN_TRAJS:-0}

t0=$(date +%s)
elapsed() { echo "[$(( $(date +%s) - t0 ))s]"; }

# -----------------------------------------------------------------------------
n_trajs() {   # n_trajs FILE -> trajectory count on stdout, 0 if unreadable
  [ -f "$1" ] || { echo 0; return; }
  TRAJ_FILE="$1" python - <<'PY' 2>/dev/null || echo 0
import os, h5py
try:
    with h5py.File(os.environ['TRAJ_FILE'], 'r') as f:
        print(len(f.keys()))
except Exception:
    print(0)
PY
}

n_source_eps() {   # n_source_eps H5PATH -> episode count from the sidecar json
  SIDE="${1%.h5}.json" python - <<'PY' 2>/dev/null || echo 0
import os, json
try:
    print(len(json.load(open(os.environ['SIDE']))['episodes']))
except Exception:
    print(0)
PY
}

# Returns 0 (reuse) if the output looks complete; otherwise moves it aside and
# returns 1 so the caller regenerates. Never silently keeps a short dataset.
reusable() {   # reusable OUTFILE
  local f=$1 n
  [ -f "$f" ] || return 1
  n=$(n_trajs "$f")
  if [ "${n:-0}" -ge "$MIN_TRAJS" ]; then
    echo "  reusing $(basename "$f")  ($n trajectories)"
    echo "  (delete it to force a re-run)"
    return 0
  fi
  echo "  found $(basename "$f") with only ${n:-0} trajectories (want >= $MIN_TRAJS)"
  echo "  -> setting it aside and regenerating"
  mv "$f" "$f.short.$(date +%s)"
  return 1
}

report_yield() {   # report_yield OUTFILE SRCFILE
  local n src
  n=$(n_trajs "$1"); src=$(n_source_eps "$2")
  if [ "${src:-0}" -gt 0 ]; then
    awk -v n="$n" -v s="$src" 'BEGIN{printf "  yield: %d/%d = %.0f%%\n", n, s, 100*n/s}'
    if [ "$n" -lt "$MIN_TRAJS" ]; then
      echo "  !! Below $MIN_TRAJS. ~97% is expected with --use-env-states, so a low"
      echo "  !! number here means the flags are wrong or the run was cut short."
      echo "  !! A thin dataset caps your T-I baseline - fix it before training."
    fi
  else
    echo "  $n trajectories (no sidecar json to compare against)"
  fi
}

echo "=============================================================="
echo "Downloading PushT-v1 demonstrations"
echo "=============================================================="
python -m mani_skill.utils.download_demo "PushT-v1"

echo ""
echo "Present in $DEMOS:"
ls -1 "$DEMOS"/*.h5 2>/dev/null | sed 's|.*/|  |' || echo "  (none)"

# -----------------------------------------------------------------------------
# Naming is trajectory.<obs_mode>.<control_mode>.<backend>.h5 . PushT ships
# PRE-REPLAYED into several control-mode variants at obs_mode=none (actions +
# env states, no observations) - there is no raw trajectory.h5 for this task.
# Discover rather than hardcode; the shipped set varies by task and release.
# -----------------------------------------------------------------------------
SRC=$(ls "$DEMOS"/trajectory.none."$CTRL".*.h5 2>/dev/null | head -1)
if [ -z "$SRC" ]; then
  SRC=$(ls "$DEMOS"/trajectory*.h5 2>/dev/null \
        | grep -vE '\.(state|rgb|rgbd|pointcloud)\.' | head -1)
  [ -n "$SRC" ] && echo "" && \
    echo "!! No source for '$CTRL'; falling back to $SRC" && \
    echo "!! Its control mode may differ from \$CTRL - check gate 9 before training."
fi
if [ -z "$SRC" ]; then
  echo ""
  echo "!! No source trajectory for control mode '$CTRL' in $DEMOS."
  echo "!! Pick one of the files listed above and re-run with e.g."
  echo "!!   CTRL=pd_ee_delta_pose bash replay_pusht.sh"
  exit 1
fi
SRC_EPS=$(n_source_eps "$SRC")
if [ "${MIN_TRAJS:-0}" -eq 0 ]; then
  if [ "${SRC_EPS:-0}" -gt 0 ]; then
    MIN_TRAJS=$(( SRC_EPS * 85 / 100 ))
  else
    MIN_TRAJS=600
    echo "!! no sidecar episode count; falling back to MIN_TRAJS=600"
  fi
fi

echo ""
echo "Source:       $SRC"
echo "Control mode: $CTRL"
echo "Episodes:     ${SRC_EPS:-?}  (completeness threshold: $MIN_TRAJS)"

STATE_OUT=$DEMOS/trajectory.state.$CTRL.$BACKEND.h5
RGB_OUT=$DEMOS/trajectory.rgb.$CTRL.$BACKEND.h5

# -----------------------------------------------------------------------------
echo ""
echo "Demo metadata (read this before training):"
SRC="$SRC" python - <<'PY'
import json, os
p = os.environ['SRC'][:-3] + '.json'
try:
    d = json.load(open(p))
except FileNotFoundError:
    print('  (no sidecar json at', p, ')'); raise SystemExit(0)
print('  env_id      :', d['env_info']['env_id'])
print('  max_steps   :', d['env_info']['max_episode_steps'])
print('  env_kwargs  :', d['env_info']['env_kwargs'])
print('  source_type :', d.get('source_type'))
print('  source_desc :', d.get('source_desc'))
print('  n_episodes  :', len(d['episodes']))
print('  control_mode:', d['episodes'][0].get('control_mode'))
print()
print('  Report note: RL-generated demos are far less multi-modal than human')
print('  teleop, so diffusion policy\'s usual edge over plain BC is muted here.')
PY

# -----------------------------------------------------------------------------
# Disk guard. The RGB dataset is the big one and running out of space midway
# leaves a truncated h5 that fails confusingly at training time.
AVAIL=$(df -BG --output=avail "$MS_ASSET_DIR" | tail -1 | tr -dc '0-9')
echo ""
echo "Free space at $MS_ASSET_DIR: ${AVAIL}G"
if [ "${AVAIL:-0}" -lt 40 ]; then
  echo "!! Under 40G free. The RGB replay will likely fill the disk."
  echo "!! Free space or lower RGB_ENVS, then re-run."
  exit 1
fi

# -----------------------------------------------------------------------------
echo ""
echo "=============================================================="
echo "STATE replay $(elapsed)"
echo "=============================================================="
# Fast. Get this working and start a state-based training run before the RGB
# pass finishes - it tells you whether the data/eval plumbing is right without
# spending GPU-hours on pixels.
#
# BACKENDS: the backend is in the output filename for a reason. PushT is precise
# enough that a 1e-3 discrepancy flips a success into a failure, so the backend
# you replay on MUST be the backend you evaluate on. physx_cuda throughout.
#
# Source is already $CTRL/$BACKEND, so this pass is additive - it attaches
# observations rather than converting controllers.
if reusable "$STATE_OUT"; then
  :
else
  python -m mani_skill.trajectory.replay_trajectory \
    --traj-path "$SRC" \
    --use-env-states \
    -c "$CTRL" -o state -b "$BACKEND" \
    --num-envs "$STATE_ENVS" --max-retry "$MAX_RETRY" --save-traj
  if [ -f "$STATE_OUT" ]; then
    report_yield "$STATE_OUT" "$SRC"
  else
    echo "  !! no output produced"
  fi
fi

# -----------------------------------------------------------------------------
echo ""
echo "=============================================================="
echo "RGB replay $(elapsed)  — this is the slow one"
echo "=============================================================="
# Much slower and much larger on disk. Lower RGB_ENVS if you OOM.
# Measured reference: ~13 min for 719 episodes at --num-envs 256.
if reusable "$RGB_OUT"; then
  :
else
  python -m mani_skill.trajectory.replay_trajectory \
    --traj-path "$SRC" \
    --use-env-states \
    -c "$CTRL" -o rgb -b "$BACKEND" \
    --num-envs "$RGB_ENVS" --max-retry "$MAX_RETRY" --save-traj
  if [ -f "$RGB_OUT" ]; then
    echo "  wrote $RGB_OUT"
    report_yield "$RGB_OUT" "$SRC"
  else
    echo "  !! no output produced"
  fi
fi

# -----------------------------------------------------------------------------
echo ""
echo "=============================================================="
echo "Done $(elapsed)"
echo "=============================================================="
du -sh "$DEMOS"/*.h5 2>/dev/null | sed 's/^/  /'

# Guard against handing you a command that names a dataset which isn't there.
NDEMOS=$(n_trajs "$STATE_OUT")
NDEMOS_RGB=$(n_trajs "$RGB_OUT")
echo ""
echo "  state: ${NDEMOS} trajectories    rgb: ${NDEMOS_RGB} trajectories"

echo ""
echo "Back these up before you terminate the pod - the RGB set is the one"
echo "artifact here that is genuinely expensive to regenerate:"
echo "  aws s3 sync $DEMOS s3://YOUR-BUCKET/pusht-data --endpoint-url https://YOUR-R2"
echo ""
echo "--------------------------------------------------------------"
echo "Before the long run, two things worth ten minutes:"
echo ""
echo "  1. --total_iters 400000 is the reference default, not a target. Find"
echo "     where the maintainers' success curve actually plateaus. If it is at"
echo "     150k your throughput problem is a third the size, for free."
echo ""
echo "  2. Sweep --num-dataload-workers 0/4/8 at --total_iters 300 with"
echo "     --eval_freq 100000. GPU ~20% while CPU ~58% is data starvation, and"
echo "     0 workers collates every batch serially in the main process."
echo "     Sweep it on the RGB dataset, not the state one - image loading"
echo "     changes both sides of the ratio."
echo "--------------------------------------------------------------"
echo ""
echo "Reference T-I training commands:"
cat <<EOF

  cd \$MANISKILL_REPO/examples/baselines/diffusion_policy

  # state-based first - plumbing and throughput, not the deliverable
  python train.py --env-id PushT-v1 \\
    --demo-path $STATE_OUT \\
    --control-mode $CTRL --sim-backend $BACKEND \\
    --num-demos $NDEMOS --max_episode_steps 150 \\
    --total_iters 400000 --log_freq 100 --eval_freq 5000 \\
    --num_eval_envs 100 --num-dataload-workers 8 --no-capture-video \\
    --exp-name dp-PushT-state --demo_type=rl --track

  # then visual (train_rgbd.py), with $RGB_OUT  ($NDEMOS_RGB demos)
  # This one is the T-I deliverable.

  Confirm flag names against \`python train.py --help\` on your version first;
  --num-dataload-workers in particular has moved between releases.
EOF
