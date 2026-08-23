#!/usr/bin/env bash
# ==============================================================================
# RIPL Lab assignment - RUNTIME verification (Gates 4-11)
#
# Scope: everything setup_runpod.sh deliberately does not check - simulation,
#        rendering, video, data pipeline, training, evaluation. Runs the whole
#        T-I pipeline at toy scale so every seam is exercised once, cheaply.
#
# The goal is NOT a good policy. It is that the expensive run later fails for
# interesting reasons rather than boring ones.
#
# Runtime: ~30 min on a 4090. Run inside tmux.
# Usage:   source /workspace/ripl/env.sh && bash smoke_test_e2e.sh
# ==============================================================================
set -uo pipefail   # NOT -e: we want to run every gate and report at the end

: "${MS_ASSET_DIR:?source /workspace/ripl/env.sh first}"
: "${MANISKILL_REPO:?source /workspace/ripl/env.sh first}"

OUT=$RIPL_ROOT/smoke
mkdir -p "$OUT"
DEMOS=$MS_ASSET_DIR/demos/PushT-v1/rl
RESULTS=()
t0=$(date +%s)

# -----------------------------------------------------------------------------
# CONTROL MODE - must match replay_pusht.sh. See the long comment there for
# why pd_ee_delta_pos: T-I's deliverable is visual (the maintainers' RGB PushT
# baselines use pos), and T-IV's residual bound alpha is only a coherent
# physical quantity when the action space is translation-only. PushT is planar.
#
# Gate 9 is what catches a mismatch between this and the replayed demos.
CTRL=${CTRL:-pd_ee_delta_pos}

# Eval env count for gate 11. The old default of 10 was a hang-debugging
# leftover. Per-step cost here is dominated by fixed Python/kernel-launch
# overhead rather than per-environment work, so 100 buys 10x the episodes for
# close to the same wall clock - gate 11 now prints s/step so you can check
# that claim on your card instead of taking it on faith.
NUM_EVAL_ENVS=${NUM_EVAL_ENVS:-100}
EVAL_SEED=${EVAL_SEED:-0}

MAX_RETRY=${MAX_RETRY:-5}
SMOKE_DEMOS=${SMOKE_DEMOS:-100}
# What counts as "the real dataset" rather than a smoke-scale one. Derived from
# the source episode count in gate 8, because PushT ships 888 / 719 / 999
# episodes for pd_ee_delta_pos / pose / pd_joint_delta_pos respectively - any
# fixed number is wrong for at least two of them. Set to override.
MIN_FULL=${MIN_FULL:-0}

gate() {   # gate <number> <name>
  echo ""
  echo "--------------------------------------------------------------"
  echo "GATE $1 — $2   [$(( $(date +%s) - t0 ))s]"
  echo "--------------------------------------------------------------"
  CURRENT="$1 $2"
}
ok()   { RESULTS+=("PASS  $CURRENT"); echo "  ✓ pass"; }
bad()  { RESULTS+=("FAIL  $CURRENT — $*"); echo "  ✗ FAIL: $*"; }
skip() { RESULTS+=("SKIP  $CURRENT — $*"); echo "  – skip: $*"; }

n_trajs() {   # n_trajs FILE -> count on stdout, 0 if missing/unreadable
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

nvidia-smi --query-gpu=memory.used --format=csv,noheader -l 5 > "$OUT/vram.log" 2>&1 &
NVSMI=$!
trap 'kill $NVSMI 2>/dev/null || true' EXIT

# =============================================================================
gate 4 "CPU sim, no rendering"
# =============================================================================
# Cheapest possible 'does the simulator run'. Isolates python-level problems
# from graphics-level ones.
if python - <<'PY'
import gymnasium as gym, mani_skill.envs
env = gym.make('PushT-v1', num_envs=1, obs_mode='state')
obs, _ = env.reset(seed=0)
print('  obs', tuple(obs.shape), '| action space', env.action_space)
obs, r, term, trunc, info = env.step(env.action_space.sample())
print('  stepped, reward', float(r))
env.close()
PY
then ok; else bad "simulator will not construct or step"; fi

# =============================================================================
gate 5 "GPU sim + rendering, and pixels are real"
# =============================================================================
# First test that exercises Vulkan through SAPIEN rather than through
# vulkaninfo. They can disagree. The mean-brightness assert catches a renderer
# that returns correctly-shaped all-zero frames - which passes every structural
# check and silently destroys visual training.
if python - <<'PY'
import gymnasium as gym, mani_skill.envs
env = gym.make('PushT-v1', num_envs=16, obs_mode='rgb', sim_backend='physx_cuda')
obs, _ = env.reset(seed=0)

def find_rgb(d):
    for k, v in d.items():
        if hasattr(v, 'shape'):
            if k == 'rgb': return v
        elif isinstance(v, dict):
            r = find_rgb(v)
            if r is not None: return r
    return None

img = find_rgb(obs)
assert img is not None, 'no rgb tensor found in observation'
print('  rgb', tuple(img.shape), img.dtype)
assert img.shape[0] == 16, f'batch dim {img.shape[0]}, expected 16'
m = img.float().mean().item()
print(f'  mean pixel value {m:.2f}')
assert m > 1.0, 'ALL-BLACK FRAMES - renderer produces no pixels'
env.close()
PY
then ok; else bad "check batch dim / black frames; try num_envs=4 to separate memory from driver"; fi

# =============================================================================
gate 6 "video writing"
# =============================================================================
# T-II needs mp4 rollouts. Headless video failure is usually silent.
rm -rf "$OUT/gate6"
python -m mani_skill.examples.demo_random_action \
  -e PushT-v1 --render-mode="rgb_array" --record-dir="$OUT/gate6" > "$OUT/gate6.log" 2>&1
MP4=$(find "$OUT/gate6" -name "*.mp4" 2>/dev/null | head -1)
if [ -n "$MP4" ] && [ "$(stat -c%s "$MP4")" -gt 10000 ]; then
  echo "  $MP4  ($(stat -c%s "$MP4") bytes)"
  ffprobe -hide_banner "$MP4" 2>&1 | grep -E "Duration|Stream" | sed 's/^/  /'
  echo "  NOTE: download and watch it. A valid 40KB mp4 of one frozen frame"
  echo "        passes this check and is still broken."
  ok
else
  bad "no usable mp4 (see $OUT/gate6.log); check ffmpeg + imageio-ffmpeg"
fi

# =============================================================================
gate 7 "throughput baseline (measurement, not pass/fail)"
# =============================================================================
python -m mani_skill.examples.benchmarking.gpu_sim -e PushT-v1 -n 256 --obs-mode state \
  2>&1 | tee "$OUT/fps_state.log" | tail -4 | sed 's/^/  /'
python -m mani_skill.examples.benchmarking.gpu_sim -e PushT-v1 -n 128 --obs-mode rgb \
  2>&1 | tee "$OUT/fps_rgb.log" | tail -4 | sed 's/^/  /'
echo "  Record both. If RGB is within 2x of state, rendering is likely falling"
echo "  back to something slow - investigate before committing to a long run."
RESULTS+=("INFO  7 throughput — see $OUT/fps_*.log")

# =============================================================================
gate 8 "data pipeline: download + small replay"
# =============================================================================
python -m mani_skill.utils.download_demo "PushT-v1" > "$OUT/download.log" 2>&1

# Naming is trajectory.<obs_mode>.<control_mode>.<backend>.h5 . PushT ships
# PRE-REPLAYED into several control-mode variants at obs_mode=none (actions +
# env states, no observations) - there is no raw trajectory.h5 for this task.
# Discover the source rather than hardcoding, since the shipped set varies by
# task and by release.
DATA=$DEMOS/trajectory.state.$CTRL.physx_cuda.h5

echo "  control mode: $CTRL"
echo "  available in $DEMOS:"
ls -1 "$DEMOS"/*.h5 2>/dev/null | sed 's|.*/|    |' || echo "    (none)"

SRC=$(ls "$DEMOS"/trajectory.none.$CTRL.*.h5 2>/dev/null | head -1)
if [ -z "$SRC" ]; then
  # fall back to any non-derived trajectory (exclude our own outputs)
  SRC=$(ls "$DEMOS"/trajectory*.h5 2>/dev/null \
        | grep -vE '\.(state|rgb|rgbd|pointcloud)\.' | head -1)
  [ -n "$SRC" ] && echo "  !! no source for '$CTRL'; falling back to $SRC"
fi

if [ -z "$SRC" ]; then
  bad "no source trajectory found in $DEMOS (see $OUT/download.log)"
  find "$MS_ASSET_DIR/demos" -name "*.h5" | sed 's/^/    /'
else
  echo "  source: $SRC"
  SRC_EPS=$(SIDE="${SRC%.h5}.json" python - <<'PY' 2>/dev/null || echo 0
import os, json
try:
    print(len(json.load(open(os.environ['SIDE']))['episodes']))
except Exception:
    print(0)
PY
)
  if [ "${MIN_FULL:-0}" -eq 0 ]; then
    if [ "${SRC_EPS:-0}" -gt 0 ]; then MIN_FULL=$(( SRC_EPS * 85 / 100 ))
    else MIN_FULL=600; fi
  fi
  echo "  episodes in source: ${SRC_EPS:-?}  (full-dataset threshold: $MIN_FULL)"
  SRC="$SRC" python - <<'PY'
import json, os
p = os.environ['SRC'][:-3] + '.json'
try:
    d = json.load(open(p))
except FileNotFoundError:
    print('  (no sidecar json at', p, ')'); raise SystemExit(0)
print('  env_kwargs  :', d['env_info']['env_kwargs'])
print('  max_steps   :', d['env_info']['max_episode_steps'])
print('  source_type :', d.get('source_type'), '  <- note for the report:')
print('                 RL-generated demos are far less multi-modal than human')
print('                 teleop, which mutes diffusion policy\'s usual edge over BC')
print('  episodes    :', len(d['episodes']))
print('  control_mode:', d['episodes'][0].get('control_mode'))
PY

  # --count 30 keeps this to a couple of minutes. Backend is baked into the
  # output filename on purpose: PushT is precise enough that a 1e-3 discrepancy
  # flips success to failure, so replay and eval backends MUST match.
  #
  # --use-env-states resets the sim to the recorded state at EVERY timestep,
  # so numerical divergence cannot accumulate through contact events. The
  # alternative, --use-first-env-state, replays open-loop from t=0 and loses
  # ~60% of PushT demos to drift. These flags come from
  # scripts/data_generation/replay_for_il_baselines.sh.
  #
  # replay_trajectory discards any episode whose replay does not succeed -
  # that filtering is ManiSkill's, not ours. --allow-failure disables it, but
  # do not: training IL on failed demonstrations is worse than having fewer.
  #
  # Reuse is BANDED, not a single threshold. A bare ">= 30" cannot tell a real
  # 694-trajectory dataset from a 212-trajectory one produced with the old
  # open-loop flag - both pass, and the short one silently caps T-I.
  EXISTING=$(n_trajs "$DATA")

  if [ "$EXISTING" -ge "$MIN_FULL" ]; then
    echo "  reusing FULL dataset: $DATA ($EXISTING trajectories)"
    echo "  (this is replay_pusht.sh's output; not overwriting it)"
    ok
  elif [ "$EXISTING" -ge 20 ]; then
    echo "  reusing SMOKE-SCALE dataset: $DATA ($EXISTING trajectories)"
    echo ""
    echo "  !! $EXISTING is enough to exercise the pipeline but is NOT a training"
    echo "  !! set. If you expected the full ~700, this file came from a capped"
    echo "  !! run or from --use-first-env-state. Delete it and run"
    echo "  !! replay_pusht.sh before trusting any T-I number."
    ok
  else
    python -m mani_skill.trajectory.replay_trajectory \
      --traj-path "$SRC" \
      --use-env-states -c "$CTRL" -o state -b physx_cuda \
      --num-envs 16 --count 30 --max-retry "$MAX_RETRY" --save-traj \
      > "$OUT/replay.log" 2>&1

    if [ -f "$DATA" ]; then
      NTRAJ=$(n_trajs "$DATA")
      echo "  wrote: $DATA  ($NTRAJ / 30 trajectories)"
      if [ "${NTRAJ:-0}" -lt 20 ]; then
        echo "  !! low yield even with --max-retry $MAX_RETRY."
        echo "  !! ~97% is expected with --use-env-states, so this points at the"
        echo "  !! flags or the source file, not at retry count. Very small"
        echo "  !! trajectory counts also stall IterationBasedBatchSampler in gate 10."
        grep -iE 'fail|skip|success|saved|replayed' "$OUT/replay.log" | tail -8 | sed 's/^/     /'
        bad "replay yield $NTRAJ/30"
      else
        ok
      fi
    else
      bad "replay produced no $DATA (see $OUT/replay.log)"
      tail -15 "$OUT/replay.log" | sed 's/^/    /'
    fi
  fi
fi

# =============================================================================
gate 9 "action-dim assertion"
# =============================================================================
# Cheapest gate, highest payoff. Known failure: replayed demos come out 4-dim
# while the eval env defaults to 7-dim. Training runs fine and yields a policy
# that cannot act. This is also what catches a stale dataset built under a
# different control mode after you switch CTRL.
if [ ! -f "$DATA" ]; then
  skip "no dataset from gate 8"
elif DATA="$DATA" CTRL="$CTRL" python - <<'PY'
import os, h5py, gymnasium as gym, mani_skill.envs
p = os.environ['DATA']
with h5py.File(p, 'r') as f:
    k = sorted(f.keys())[0]
    print('  demo actions', f[k]['actions'].shape, '| obs', f[k]['obs'].shape,
          f'| {len(f.keys())} trajs')
    demo_dim = f[k]['actions'].shape[-1]
env = gym.make('PushT-v1', num_envs=1, obs_mode='state',
               control_mode=os.environ['CTRL'], sim_backend='physx_cuda')
env_dim = env.action_space.shape[-1]; env.close()
print(f'  demo {demo_dim} vs env {env_dim} (control_mode={os.environ["CTRL"]})')
assert demo_dim == env_dim, (
    'MISMATCH - the dataset was replayed under a different control mode than '
    'CTRL. Fix the replay -c flag and regenerate; do not "fix" the eval.')
PY
then ok; else bad "action dim mismatch between demos and env"; fi

# =============================================================================
gate 10 "training slice, 2000 iters"
# =============================================================================
# Exercises dataloader, model, optimizer, checkpointing, eval loop, VRAM.
# Success rate will be ~0. That is correct: a capped demo count and 2k iters is
# plumbing, not a policy. Do NOT tune anything based on this number.
TRAIN_RC=1; TRAIN_SECS=0
DP_DIR="$MANISKILL_REPO/examples/baselines/diffusion_policy"
if [ ! -f "$DATA" ]; then
  skip "no dataset from gate 8"
  RESULTS+=("SKIP  11 eval harness — not blocked, but train.py never ran")
elif [ ! -d "$DP_DIR" ]; then
  # Previously this was `cd ... || bad`, which reported the failure and then
  # carried on running train.py from whatever directory we happened to be in.
  bad "baseline dir missing: $DP_DIR"
else
  pushd "$DP_DIR" > /dev/null || exit 1
  echo "  available flags on YOUR mani_skill version:"
  python train.py --help 2>&1 | head -40 | sed 's/^/    /'
  echo ""
  t_train=$(date +%s)
  # Derive from the file: the replay can yield fewer trajectories than --count
  # requested, and train.py hard-asserts if --num-demos exceeds what's there.
  NTRAJ=$(n_trajs "$DATA")
  [ "${NTRAJ:-0}" -lt 1 ] && NTRAJ=8
  # Cap the smoke-test training set. On a full 694-trajectory dataset there is
  # no reason for a plumbing test to touch all of it; the evals dominate
  # runtime anyway. Raise SMOKE_DEMOS if you want a longer check.
  [ "$NTRAJ" -gt "$SMOKE_DEMOS" ] && NTRAJ=$SMOKE_DEMOS
  echo "  training on $NTRAJ trajectories"
  python train.py --env-id PushT-v1 \
    --demo-path "$DATA" \
    --control-mode "$CTRL" --sim-backend physx_cuda \
    --num-demos "$NTRAJ" --max_episode_steps 150 \
    --total_iters 2000 --log_freq 100 --eval_freq 1000 \
    --num_eval_envs 10 --capture-video \
    --exp-name gate10 --demo_type=rl > "$OUT/train.log" 2>&1
  TRAIN_RC=$?
  TRAIN_SECS=$(( $(date +%s) - t_train ))
  tail -20 "$OUT/train.log" | sed 's/^/  /'
  if [ $TRAIN_RC -eq 0 ]; then
    echo "  wall clock: ${TRAIN_SECS}s for 2000 iters"
    awk -v s="$TRAIN_SECS" 'BEGIN{
      if (s>0) printf "  ~%.2f it/s -> %.0f h for 400k iters\n", 2000/s, 400000/(2000/s)/3600 }'
    echo "  If that projection is unacceptable, the lever is"
    echo "  --num-dataload-workers (see replay_pusht.sh), and the prior"
    echo "  question is whether 400k is where the curve actually plateaus."
    ok
  else
    bad "training exited $TRAIN_RC — diff the flags above against the command (see $OUT/train.log)"
  fi

  RUNDIR=$(find "$DP_DIR/runs" -maxdepth 2 -name "*gate10*" -type d 2>/dev/null | head -1)
  echo "  artifacts:"
  find "${RUNDIR:-.}" \( -name "*.pt" -o -name "*.mp4" \) 2>/dev/null | head -10 | sed 's/^/    /'
  popd > /dev/null || true
fi

# =============================================================================
gate 11 "standalone eval harness + initial-state reproducibility"
# =============================================================================
# T-II and T-IV both need to score a policy outside train.py under an
# initial-state distribution YOU control. ignore_terminations +
# reconfiguration_freq=1 are the standard ManiSkill settings; success_once is
# the metric.
#
# The reproducibility block is the part that matters for T-II. That task's
# whole method is: log initial states, find the failure region R, then resample
# FRESH episodes from R and re-measure. That requires (a) seed -> initial state
# to be deterministic, and (b) envs within a batch to draw distinct states.
# Neither is worth assuming, and both change when you change num_envs - which
# is exactly what we are doing here by going from 10 to $NUM_EVAL_ENVS.
if NUM_EVAL_ENVS="$NUM_EVAL_ENVS" EVAL_SEED="$EVAL_SEED" CTRL="$CTRL" python - <<'PY'
import os, time, gymnasium as gym, torch, mani_skill.envs
from collections import defaultdict
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

N = int(os.environ['NUM_EVAL_ENVS'])
SEED = int(os.environ['EVAL_SEED'])
CTRL = os.environ.get('CTRL', 'pd_ee_delta_pos')

envs = gym.make('PushT-v1', num_envs=N, obs_mode='state',
                control_mode=CTRL, sim_backend='physx_cuda',
                reconfiguration_freq=1)
envs = ManiSkillVectorEnv(envs, ignore_terminations=True, record_metrics=True)

def snapshot():
    """Flat tensor of the full sim state, one row per env."""
    base = envs.unwrapped
    try:
        d = base.get_state_dict()
        flat = []
        def walk(x):
            if isinstance(x, dict):
                for k in sorted(x): walk(x[k])
            else:
                flat.append(torch.as_tensor(x).reshape(N, -1).float().cpu())
        walk(d)
        return torch.cat(flat, dim=1)
    except Exception:
        return torch.as_tensor(base.get_state()).reshape(N, -1).float().cpu()

# --- (a) same seed -> same initial states --------------------------------
envs.reset(seed=SEED); a = snapshot()
envs.reset(seed=SEED); b = snapshot()
same = torch.allclose(a, b, atol=1e-6)
print(f'  reset(seed={SEED}) twice -> identical: {same}')
assert same, ('seed does not determine initial state. T-II cannot resample a '
              'failure region you cannot reproduce.')

# --- (b) envs within one batch are distinct ------------------------------
uniq = torch.unique(a, dim=0).shape[0]
print(f'  distinct initial states across {N} envs: {uniq}')
assert uniq > max(2, N // 2), (
    f'only {uniq} distinct states in {N} envs - envs are not being randomised '
    'independently, so your "100 rollouts" is really far fewer.')

# --- (c) a different seed gives a different distribution -----------------
envs.reset(seed=SEED + 1); c = snapshot()
print(f'  reset(seed={SEED+1}) differs from seed={SEED}: '
      f'{not torch.allclose(a, c, atol=1e-6)}')

# --- rollout + per-step timing -------------------------------------------
# The claim being tested: per-step cost is dominated by fixed Python/kernel
# overhead, so N=100 costs about what N=10 did. Compare this s/step against a
# run with NUM_EVAL_ENVS=10 - if they are close, the claim holds on your card.
obs, _ = envs.reset(seed=SEED)
m = defaultdict(list)
t = time.time()
STEPS = 150
for _ in range(STEPS):
    obs, r, term, trunc, info = envs.step(envs.action_space.sample())
    if trunc.any():
        for k, v in info['final_info']['episode'].items():
            m[k].append(v.float())
dt = time.time() - t

for k in m:
    print(f'  {k}: {torch.mean(torch.stack(m[k])).item():.3f}')
print(f'  {dt:.1f}s for {STEPS} steps x {N} envs '
      f'= {dt/STEPS*1000:.0f} ms/step, {dt/N*1000:.0f} ms/episode')
if torch.cuda.is_available():
    print(f'  peak VRAM this gate: '
          f'{torch.cuda.max_memory_allocated()/1e9:.2f} GB')
envs.close()
assert 'success_once' in m, 'success_once not recorded - check record_metrics'
print('  (values near 0 expected - random policy)')
PY
then ok; else bad "eval harness broken, or initial states are not reproducible / not independent"; fi

# =============================================================================
kill $NVSMI 2>/dev/null || true
echo ""
echo "=============================================================="
echo "SUMMARY   (total $(( $(date +%s) - t0 ))s)"
echo "=============================================================="
printf '  %s\n' "${RESULTS[@]}"
echo ""
echo "  control mode: $CTRL"
echo "  eval envs:    $NUM_EVAL_ENVS"
echo "  peak VRAM:    $(sort -n -r "$OUT/vram.log" 2>/dev/null | head -1)"
echo "  logs:         $OUT"
echo ""
if printf '%s\n' "${RESULTS[@]}" | grep -q '^FAIL'; then
  echo "  Red gates above. See FIRST_PASS.md for per-gate diagnosis."
  exit 1
fi
echo "  All green. Scale up:"
echo "    - bash replay_pusht.sh   (full state + rgb replay)"
echo "    - sweep --num-dataload-workers before committing to 400k iters"
echo "    - --num_eval_envs 100 --track"
echo "    - then run Gate 12 (rehydrate) from FIRST_PASS.md"
echo "=============================================================="
