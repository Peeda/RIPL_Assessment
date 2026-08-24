#!/usr/bin/env bash
# ==============================================================================
# smoke_test.sh - runtime verification. Four gates, ~3 min.
#
#   source /workspace/ripl/env.sh && bash smoke_test.sh
#
# Scope: everything setup_runpod.sh deliberately does not check, minus anything
# run_pipeline.sh already proves by doing it for real. What is left is the four
# questions whose answers are silent when wrong:
#
#   1. does the simulator construct and step
#   2. does the renderer produce actual pixels, or correctly-shaped black
#   3. does the replayed dataset's action dim match the env you will evaluate in
#   4. does a seed determine the initial state, and are envs independent
#
# Gate 4 is T-II's prerequisite. Gate 2 is the one that has saved the most time.
#
# The predecessor had eight gates; four of them (video, throughput benchmark,
# download+replay, a 2000-iteration training slice) duplicated work
# run_pipeline.sh does at full scale, and their measurements were misleading -
# see notes/pusht-detour.md.
# ==============================================================================
set -uo pipefail   # NOT -e: we want to run every gate and report at the end

: "${MS_ASSET_DIR:?source /workspace/ripl/env.sh first}"
: "${MANISKILL_REPO:?source /workspace/ripl/env.sh first}"

ENV_ID=${ENV_ID:-StackCube-v1}
CTRL=${CTRL:-pd_ee_delta_pos}
BACKEND=${BACKEND:-physx_cpu}
NUM_EVAL_ENVS=${NUM_EVAL_ENVS:-100}
EVAL_SEED=${EVAL_SEED:-0}

OUT=${RIPL_ROOT:-/workspace/ripl}/smoke
mkdir -p "$OUT"
DEMOS=$MS_ASSET_DIR/demos/$ENV_ID/motionplanning
DATA=$DEMOS/trajectory.state.$CTRL.$BACKEND.h5

RESULTS=()
t0=$(date +%s)

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

nvidia-smi --query-gpu=memory.used --format=csv,noheader -l 5 > "$OUT/vram.log" 2>&1 &
NVSMI=$!
trap 'kill $NVSMI 2>/dev/null || true' EXIT

# =============================================================================
gate 1 "sim constructs and steps"
# =============================================================================
# Cheapest possible 'does the simulator run'. Isolates python-level problems
# from graphics-level ones.
if ENV_ID="$ENV_ID" python - <<'PY'
import os, gymnasium as gym, mani_skill.envs
env = gym.make(os.environ['ENV_ID'], num_envs=1, obs_mode='state')
obs, _ = env.reset(seed=0)
print('  obs', tuple(obs.shape), '| action space', env.action_space)
obs, r, term, trunc, info = env.step(env.action_space.sample())
print('  stepped, reward', float(r))
env.close()
PY
then ok; else bad "simulator will not construct or step"; fi

# =============================================================================
gate 2 "renderer produces real pixels"
# =============================================================================
# First test that exercises Vulkan through SAPIEN rather than through
# vulkaninfo. They can disagree. The mean-brightness assert catches a renderer
# that returns correctly-shaped all-zero frames - which passes every structural
# check and silently destroys visual training, which is the T-I deliverable.
if ENV_ID="$ENV_ID" BACKEND="$BACKEND" python - <<'PY'
import os, gymnasium as gym, mani_skill.envs
N = 4
env = gym.make(os.environ['ENV_ID'], num_envs=N, obs_mode='rgb',
               sim_backend=os.environ['BACKEND'])
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
assert img.shape[0] == N, f'batch dim {img.shape[0]}, expected {N}'
m = img.float().mean().item()
print(f'  mean pixel value {m:.2f}')
assert m > 1.0, 'ALL-BLACK FRAMES - renderer produces no pixels'
env.close()
PY
then ok; else bad "black frames or bad batch dim; try num_envs=1 to separate memory from driver"; fi

# =============================================================================
gate 3 "dataset shape and action dim"
# =============================================================================
# Cheapest gate, highest payoff. A dataset replayed under a different control
# mode than you evaluate under trains fine and yields a policy that cannot act.
# Also reports what a separate dataset-audit script used to: trajectory count,
# obs presence, size, and the control mode named in the filename.
if [ ! -f "$DATA" ]; then
  skip "no dataset yet — run 'bash run_pipeline.sh data'"
elif DATA="$DATA" CTRL="$CTRL" ENV_ID="$ENV_ID" BACKEND="$BACKEND" python - <<'PY'
import os, h5py, gymnasium as gym, mani_skill.envs
p, ctrl = os.environ['DATA'], os.environ['CTRL']

named = os.path.basename(p).split('.')
print(f"  file        : {os.path.basename(p)}")
print(f"  named as    : obs_mode={named[1]} control_mode={named[2]} backend={named[3]}")
print(f"  size        : {os.path.getsize(p)/1e9:.2f} GB")
with h5py.File(p, 'r') as f:
    keys = sorted(f.keys())
    k = keys[0]
    print(f"  trajectories: {len(keys)}")
    print(f"  actions     : {f[k]['actions'].shape}")
    assert 'obs' in f[k], "no 'obs' group — replay did not attach observations"
    demo_dim = f[k]['actions'].shape[-1]

env = gym.make(os.environ['ENV_ID'], num_envs=1, obs_mode='state',
               control_mode=ctrl, sim_backend=os.environ['BACKEND'])
env_dim = env.action_space.shape[-1]; env.close()
print(f'  demo {demo_dim} vs env {env_dim} (control_mode={ctrl})')
assert demo_dim == env_dim, (
    'MISMATCH - the dataset was replayed under a different control mode than '
    'CTRL. Fix the replay -c flag and regenerate; do not "fix" the eval.')

# 4, not 3: the Panda's pd_ee_delta_pos is 3 translation + 1 gripper. A 3 here
# means the control mode is not what you think it is. The gripper dim is also
# why T-IV's residual must be applied to the translation components only.
if ctrl == 'pd_ee_delta_pos' and demo_dim != 4:
    print(f'  !! expected 4 dims (3 translation + gripper), got {demo_dim}')
PY
then ok; else bad "action dim mismatch, missing obs, or unreadable dataset"; fi

# =============================================================================
gate 4 "initial-state reproducibility"
# =============================================================================
# T-II and T-IV both need to score a policy outside train.py under an
# initial-state distribution YOU control. ignore_terminations +
# reconfiguration_freq=1 are the standard ManiSkill settings; success_once is
# the metric.
#
# The reproducibility block is the part that matters. T-II's whole method is:
# log initial states, find the failure region R, then resample FRESH episodes
# from R and re-measure. That requires (a) seed -> initial state to be
# deterministic and (b) envs within a batch to draw distinct states. Neither is
# worth assuming, and both change when you change num_envs.
#
# The timing loop is an open question again: the "100 eval envs is nearly free"
# result in notes/pusht-detour.md was measured on physx_cuda and does not
# transfer to CPU simulation. Measure it here rather than assuming it.
if NUM_EVAL_ENVS="$NUM_EVAL_ENVS" EVAL_SEED="$EVAL_SEED" CTRL="$CTRL" \
   ENV_ID="$ENV_ID" BACKEND="$BACKEND" python - <<'PY'
import os, time, gymnasium as gym, torch, mani_skill.envs
from collections import defaultdict
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

N = int(os.environ['NUM_EVAL_ENVS'])
SEED = int(os.environ['EVAL_SEED'])

envs = gym.make(os.environ['ENV_ID'], num_envs=N, obs_mode='state',
                control_mode=os.environ['CTRL'],
                sim_backend=os.environ['BACKEND'],
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
obs, _ = envs.reset(seed=SEED)
m = defaultdict(list)
t = time.time()
STEPS = 100
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
echo "  env id:       $ENV_ID"
echo "  control mode: $CTRL"
echo "  backend:      $BACKEND"
echo "  eval envs:    $NUM_EVAL_ENVS"
echo "  peak VRAM:    $(sort -n -r "$OUT/vram.log" 2>/dev/null | head -1)"
echo ""
if printf '%s\n' "${RESULTS[@]}" | grep -q '^FAIL'; then
  exit 1
fi
echo "  All green. Next: bash run_pipeline.sh data, then train."
echo ""
echo "  Not checked here: mp4 writing. --capture-video defaults to True, so"
echo "  the first training run proves it. T-II needs those videos - look at"
echo "  one rather than trusting that the file exists."
echo "=============================================================="
