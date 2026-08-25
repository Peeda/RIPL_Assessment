#!/usr/bin/env bash
# ==============================================================================
# setup/transfer.sh - work out how to get files off this pod, and print the commands.
#
#   bash setup/transfer.sh check     # can this pod do rsync at all?
#   bash setup/transfer.sh info      # paste-ready pull/push commands with IP+port
#   bash setup/transfer.sh send PATH # runpodctl fallback, no open ports needed
#
# Exists because RunPod has two SSH paths and only one moves files. The proxied
# path (ssh.runpod.io) is available on every pod but does NOT support SCP/SFTP,
# so rsync over it fails with an error that does not say that. Direct TCP SSH
# needs a public IP and an exposed port 22 - present on the official PyTorch
# template, absent on many Community Cloud shapes.
#
# Find that out now, not when you have a checkpoint you need off the box.
# ==============================================================================
set -uo pipefail

IP=${RUNPOD_PUBLIC_IP:-}
PORT=${RUNPOD_TCP_PORT_22:-}
POD=${RUNPOD_POD_ID:-unknown}

DEMOS="${MS_ASSET_DIR:-/workspace/ripl/maniskill_data}/demos"
RUNS="${MANISKILL_REPO:-/workspace/ripl/ManiSkill}/examples/baselines/diffusion_policy/runs"
SMOKE="${RIPL_ROOT:-/workspace/ripl}/smoke"
# T-II mining output: CSVs, traces, init/final sim states, mp4s. Small, but the
# only irreplaceable thing here besides the checkpoints - a mined pass is hours
# of GPU time and cannot be regenerated, because the rollouts are stochastic.
T2="${T2_OUT:-${RIPL_ROOT:-/workspace/ripl}/t2}"

case "${1:-info}" in

check)
  echo "pod: $POD"
  if [ -n "$IP" ] && [ -n "$PORT" ]; then
    echo "direct TCP SSH: yes   ($IP:$PORT)"
    pgrep -x sshd >/dev/null \
      && echo "sshd running:   yes" \
      || echo "sshd running:   NO  -> start it: service ssh start"
    echo ""
    echo "rsync will work. Run 'bash setup/transfer.sh info' for the commands."
  else
    echo "direct TCP SSH: NO"
    echo ""
    echo "RUNPOD_PUBLIC_IP / RUNPOD_TCP_PORT_22 are unset, so this pod only has"
    echo "the ssh.runpod.io proxy - which cannot do SCP, SFTP or rsync."
    echo ""
    echo "Options, in order of preference:"
    echo "  1. Terminate and redeploy on a shape with a public IP (Secure Cloud,"
    echo "     official PyTorch template). Do this NOW if you intend to train"
    echo "     here - discovering it later costs you the run."
    echo "  2. Use runpodctl instead:  bash setup/transfer.sh send /path/to/file"
    exit 1
  fi
  ;;

info)
  if [ -z "$IP" ] || [ -z "$PORT" ]; then
    echo "!! No direct TCP SSH on this pod. Run 'bash setup/transfer.sh check'."
    exit 1
  fi
  R="rsync -avzP -e \"ssh -p $PORT\""
  cat <<EOF
Pod $POD at $IP:$PORT

  -P is --partial --progress: an interrupted 400 MB transfer resumes rather
  than starting over. Run these ON YOUR LAPTOP, not here.

PULL (pod -> laptop)

  # datasets - saves ~20 min of replay when you rebuild
  $R root@$IP:$DEMOS/ ~/ripl/demos/

  # checkpoints - the irreplaceable half. Pull when a RUN ends.
  $R root@$IP:$RUNS/ ~/ripl/runs/

  # videos and logs for the report
  $R root@$IP:$SMOKE/ ~/ripl/smoke/

  # T-II mining results - rollout CSVs, traces, mp4s. Pull when a PASS ends,
  # not when the night does.
  $R root@$IP:$T2/ ~/ripl/t2/

PUSH (laptop -> fresh pod, after setup_runpod.sh)

  $R ~/ripl/demos/ root@$IP:$DEMOS/
  $R ~/ripl/runs/  root@$IP:$RUNS/

  Then verify - a completed rsync is not a complete dataset:
    bash setup/smoke_test.sh

DRY RUN anything you are unsure about by adding -n.
EOF
  ;;

send)
  SRC=${2:-}
  [ -n "$SRC" ] || { echo "usage: bash setup/transfer.sh send /path/to/file"; exit 1; }
  [ -e "$SRC" ] || { echo "!! no such path: $SRC"; exit 1; }
  command -v runpodctl >/dev/null || {
    echo "!! runpodctl not installed:"
    echo "   wget -qO- cli.runpod.net | sudo bash"
    exit 1; }
  # runpodctl sends one file, so pack directories first.
  if [ -d "$SRC" ]; then
    TGZ="/tmp/$(basename "$SRC").tgz"
    echo "packing $SRC -> $TGZ"
    tar czf "$TGZ" -C "$(dirname "$SRC")" "$(basename "$SRC")" || exit 1
    SRC="$TGZ"
  fi
  echo "size: $(du -h "$SRC" | cut -f1)"
  echo "Run the printed 'runpodctl receive' command on your laptop."
  echo "This blocks until the transfer completes - keep the terminal open."
  echo ""
  runpodctl send "$SRC"
  ;;

*)
  sed -n '2,16p' "$0"
  exit 1
  ;;
esac
