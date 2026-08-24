#!/usr/bin/env bash
# ==============================================================================
# run_pipeline.sh - StackCube-v1 data + diffusion policy training, end to end.
#
#   source /workspace/ripl/env.sh
#   tmux new -s ripl
#   bash run_pipeline.sh data 2>&1 | tee ~/data.log
#
# Stages:  data | train | train-rgb | all   (default: all)
#
# Every command here is ManiSkill's documented recipe. Nothing is tuned, and
# nothing should be tuned without a reason written down in CLAUDE.md.
#
# The predecessor of this file was 311 lines, almost all of it absorbing the
# fact that PushT ships pre-replayed with a different episode count per control
# mode. StackCube ships one raw trajectory.h5. There is nothing left to guard.
# ==============================================================================
set -euo pipefail

ENVSH="${RIPL_ROOT:-/workspace/ripl}/env.sh"
# shellcheck source=/dev/null
[ -f "$ENVSH" ] && source "$ENVSH"
: "${MS_ASSET_DIR:?source /workspace/ripl/env.sh first}"
: "${MANISKILL_REPO:?source /workspace/ripl/env.sh first}"

# ENV_ID is a variable because this repo has already changed task once. Every
# command below is task-parametric; nothing else needs to move.
ENV_ID=${ENV_ID:-StackCube-v1}
CTRL=pd_ee_delta_pos       # 4-dim on the Panda: 3 translation + 1 gripper
BACKEND=physx_cpu
NUM_DEMOS=${NUM_DEMOS:-100}

# --- verified against examples/baselines/diffusion_policy/baselines.sh -------
# StackCube's two baseline lines do NOT share total_iters: state trains for
# 30k, rgb for 100k. Do not collapse these into one variable.
#
# max_episode_steps 200 against a registered default of 50. The README's rule
# is ~2x mean demo length, and motionplanning demos are slow.
MAX_EP_STEPS=${MAX_EP_STEPS:-200}
STATE_ITERS=${STATE_ITERS:-30000}
RGB_ITERS=${RGB_ITERS:-100000}

DEMOS=$MS_ASSET_DIR/demos/$ENV_ID/motionplanning
STATE_H5=$DEMOS/trajectory.state.$CTRL.$BACKEND.h5
RGB_H5=$DEMOS/trajectory.rgb.$CTRL.$BACKEND.h5
DP_DIR=$MANISKILL_REPO/examples/baselines/diffusion_policy

t0=$(date +%s)
stage() { echo ""; echo "=== $* [$(( $(date +%s) - t0 ))s] ==="; }

# --track fails at startup without a wandb login. Check before a long stage,
# not after one. Checks credentials directly rather than via a wandb subcommand,
# since the CLI's command set moves between versions and a false negative here
# would block the run for no reason.
need_wandb() {
  [ -n "${WANDB_API_KEY:-}" ] && return 0
  grep -qs 'api\.wandb\.ai' "$HOME/.netrc" && return 0
  echo "!! wandb not logged in. Run 'wandb login' - --track fails at startup."
  exit 1
}

do_data() {
  stage "demos: $ENV_ID"
  # StackCube-v1 is in download_demo.py's DATASET_SOURCES, so the download is
  # the expected path. The motion planner is the fallback for a task that is
  # not hosted; which one ran is worth knowing, so say so.
  if [ -f "$DEMOS/trajectory.h5" ]; then
    echo "reusing $DEMOS/trajectory.h5"
  elif python -m mani_skill.utils.download_demo "$ENV_ID"; then
    echo "downloaded"
  else
    echo "no hosted demos; generating with the motion planner"
    python -m mani_skill.examples.motionplanning.panda.run -e "$ENV_ID" --num-traj 1000
  fi

  # --use-first-env-state, NOT --use-env-states. Replay is converting the
  # controller (motionplanning records pd_joint_pos), so it has to simulate
  # forward from the initial state. See CLAUDE.md; this reverses the PushT rule.
  stage "state replay"
  [ -f "$STATE_H5" ] && echo "reusing $STATE_H5" || \
    python -m mani_skill.trajectory.replay_trajectory \
      --traj-path "$DEMOS/trajectory.h5" \
      --use-first-env-state -c "$CTRL" -o state \
      --save-traj --num-envs 10 -b "$BACKEND"

  stage "rgb replay (the slow one)"
  [ -f "$RGB_H5" ] && echo "reusing $RGB_H5" || \
    python -m mani_skill.trajectory.replay_trajectory \
      --traj-path "$DEMOS/trajectory.h5" \
      --use-first-env-state -c "$CTRL" -o rgb \
      --save-traj --num-envs 10 -b "$BACKEND"

  stage "done"
  du -sh "$DEMOS"/*.h5 2>/dev/null | sed 's/^/  /'
}

do_train() {
  need_wandb
  stage "state training - plumbing and throughput, not the deliverable"
  cd "$DP_DIR"
  python train.py --env-id "$ENV_ID" \
    --demo-path "$STATE_H5" \
    --control-mode "$CTRL" --sim-backend "$BACKEND" \
    --num-demos "$NUM_DEMOS" --max_episode_steps "$MAX_EP_STEPS" \
    --total_iters "$STATE_ITERS" \
    --demo_type=motionplanning --track \
    --exp-name "diffusion_policy-$ENV_ID-state-${NUM_DEMOS}_motionplanning_demos-1"
}

do_train_rgb() {
  need_wandb
  stage "rgb training - this checkpoint is the T-I deliverable"
  cd "$DP_DIR"
  python train_rgbd.py --env-id "$ENV_ID" \
    --demo-path "$RGB_H5" \
    --control-mode "$CTRL" --sim-backend "$BACKEND" \
    --num-demos "$NUM_DEMOS" --max_episode_steps "$MAX_EP_STEPS" \
    --total_iters "$RGB_ITERS" --obs-mode "rgb" \
    --demo_type=motionplanning --track \
    --exp-name "diffusion_policy-$ENV_ID-rgb-${NUM_DEMOS}_motionplanning_demos-1"
}

case "${1:-all}" in
  data)      do_data ;;
  train)     do_train ;;
  train-rgb) do_train_rgb ;;
  all)       do_data; do_train; do_train_rgb ;;
  *) sed -n '2,12p' "$0"; exit 1 ;;
esac
