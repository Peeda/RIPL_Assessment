#!/usr/bin/env bash
# ==============================================================================
# setup/smoke_test.sh - runtime verification. Five gates, ~3 min.
#
#   source /workspace/ripl/env.sh && bash setup/smoke_test.sh
#
# Scope: everything setup_runpod.sh deliberately does not check, minus anything
# run_pipeline.sh already proves by doing it for real. What is left is the five
# questions whose answers are silent when wrong:
#
#   1. does the simulator construct and step
#   2. does the renderer produce actual pixels, or correctly-shaped black
#   3. does the replayed dataset's action dim match the env you will evaluate in
#   4. does a seed determine the initial state
#   5. does the subprocess eval path train.py uses actually start
#
# Gate 4 is T-II's prerequisite. Gate 2 is the one that has saved the most time.
#
# NOTE physx_cpu raises RuntimeError for num_envs > 1 - it vectorises by
# subprocess, not by batching. Every gate here is single-env except 5, which
# goes through gym.vector.AsyncVectorEnv exactly as training does.
#
# The predecessor had eight gates; four of them (video, throughput benchmark,
# download+replay, a 2000-iteration training slice) duplicated work
# run_pipeline.sh does at full scale, and their measurements were misleading -
# see notes/pusht-detour.md.
# ==============================================================================
set -uo pipefail   # NOT -e: we want to run every gate and report at the end

# Source env.sh ourselves, as run_pipeline.sh does, so either script works from
# a bare shell. This is not just convenience: env.sh sets MS_ASSET_DIR to the
# /workspace path, and without it ManiSkill falls back to ~/.maniskill on the
# container disk, which does not survive a pod stop. The :? assertions below
# mean a missing env.sh stops us before anything touches disk.
ENVSH="${RIPL_ROOT:-/workspace/ripl}/env.sh"
# shellcheck source=/dev/null
[ -f "$ENVSH" ] && source "$ENVSH"
: "${MS_ASSET_DIR:?source /workspace/ripl/env.sh first}"
: "${MANISKILL_REPO:?source /workspace/ripl/env.sh first}"

ENV_ID=${ENV_ID:-StackCube-v1}
CTRL=${CTRL:-pd_ee_delta_pos}
BACKEND=${BACKEND:-physx_cpu}
# physx_cpu vectorises by SUBPROCESS, not by batching. train.py's default is 10
# and that is what the eval path will actually spawn; 100 here would be 100
# processes. Match train.py rather than the old physx_cuda number.
NUM_EVAL_ENVS=${NUM_EVAL_ENVS:-10}
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
# num_envs=1: physx_cpu raises RuntimeError for anything more. CPU vectorisation
# is by subprocess (gym.vector.AsyncVectorEnv), which gate 5 exercises.
N = 1
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
then ok; else bad "black frames, or the renderer will not start headless"; fi

# =============================================================================
gate 3 "datasets: action dim, and RGB frames are not black"
# =============================================================================
# Cheapest gate, highest payoff. Checks BOTH datasets, because they fail
# differently:
#
#   state: a wrong -c at replay time gives a dataset whose action dim does not
#          match the eval env. Training runs and yields a policy that cannot act.
#
#   rgb:   if the renderer was not working when the replay ran, the h5 contains
#          correctly-shaped ALL-BLACK images. Training runs, loss decreases (the
#          net learns the mean action from a constant input), and success
#          flatlines near zero forever. Gate 2 catches a dead renderer live;
#          nothing caught it in the stored dataset until now.
STATE_H5=$DEMOS/trajectory.state.$CTRL.$BACKEND.h5
RGB_H5=$DEMOS/trajectory.rgb.$CTRL.$BACKEND.h5
if [ ! -f "$STATE_H5" ] && [ ! -f "$RGB_H5" ]; then
  skip "no datasets yet — run 'bash t1/run_pipeline.sh data'"
elif STATE_H5="$STATE_H5" RGB_H5="$RGB_H5" CTRL="$CTRL" ENV_ID="$ENV_ID" \
     BACKEND="$BACKEND" python - <<'PY'
import os, h5py, numpy as np, gymnasium as gym, mani_skill.envs

env = gym.make(os.environ['ENV_ID'], num_envs=1, obs_mode='state',
               control_mode=os.environ['CTRL'], sim_backend=os.environ['BACKEND'])
env_dim = env.action_space.shape[-1]; env.close()
print(f"  env action_dim under {os.environ['CTRL']}: {env_dim}")
# 4, not 3: the Panda's pd_ee_delta_pos is 3 translation + 1 gripper. A 3 means
# the control mode is not what you think it is. That gripper dim is also why
# T-IV's residual is applied to the translation components only.
if os.environ['CTRL'] == 'pd_ee_delta_pos' and env_dim != 4:
    print(f'  !! expected 4 (3 translation + gripper), got {env_dim}')

def rgb_arrays(g, out, prefix=''):
    """Every dataset named 'rgb' anywhere under this group."""
    for k, v in g.items():
        p = f'{prefix}/{k}'
        if isinstance(v, h5py.Group):
            rgb_arrays(v, out, p)
        elif k == 'rgb':
            out.append((p, v))

problems = []
for label in ('STATE_H5', 'RGB_H5'):
    path = os.environ[label]
    name = os.path.basename(path)
    if not os.path.exists(path):
        print(f'\n  {name}: MISSING (skipped)')
        continue
    print(f'\n  {name}  ({os.path.getsize(path)/1e9:.2f} GB)')
    with h5py.File(path, 'r') as f:
        keys = sorted(f.keys())
        t = f[keys[0]]
        print(f'    trajectories : {len(keys)}')
        print(f'    actions      : {t["actions"].shape}')
        if 'obs' not in t:
            problems.append(f'{name}: no obs group - replay attached no observations')
            continue
        dim = t['actions'].shape[-1]
        if dim != env_dim:
            problems.append(
                f'{name}: action dim {dim} != env {env_dim}. The dataset was '
                f'replayed under a different control mode. Fix the replay -c '
                f'flag and regenerate; do not "fix" the eval.')

        imgs = []
        rgb_arrays(t['obs'], imgs)
        if not imgs:
            print('    observations : state vectors (no cameras)')
            continue
        for p, arr in imgs:
            frame = np.asarray(arr[0], dtype=np.float32)
            m = float(frame.mean())
            print(f'    camera {p:<34} {tuple(arr.shape)} {arr.dtype}  mean={m:.2f}')
            if m <= 1.0:
                problems.append(
                    f'{name}: {p} is ALL BLACK (mean {m:.2f}). The renderer was '
                    f'not producing pixels when this replay ran. Training on it '
                    f'gives a flat success curve forever. Delete and re-replay.')

print('')
if problems:
    for p in problems:
        print(f'  >> {p}')
    raise SystemExit(1)
print('  both datasets consistent with the eval env')
PY
then ok; else bad "dataset problem — read the >> lines above"; fi

# =============================================================================
gate 4 "initial-state reproducibility (T-II prerequisite)"
# =============================================================================
# T-II's whole method is: log initial states, find the failure region R, then
# resample FRESH episodes from R and re-measure. That needs seed -> initial
# state to be a deterministic, injective-enough map.
#
# Under physx_cpu this is a SINGLE env queried across seeds, not one batch of
# many. That is not a weaker test - it is the right one. CPU vectorisation runs
# N independent processes each with its own seed, so "are envs in a batch
# distinct" reduces to "do distinct seeds give distinct states", which is
# exactly what is asserted below and exactly what T-II resamples on.
#
# Also records the cube geometry the failure characterisation is built from.
# See CLAUDE.md for the wrap convention: all angles to (-pi, pi], at log time.
if EVAL_SEED="$EVAL_SEED" CTRL="$CTRL" ENV_ID="$ENV_ID" BACKEND="$BACKEND" \
   python - <<'PY'
import os, math, time, gymnasium as gym, torch, mani_skill.envs

SEED = int(os.environ['EVAL_SEED'])
env = gym.make(os.environ['ENV_ID'], num_envs=1, obs_mode='state',
               control_mode=os.environ['CTRL'],
               sim_backend=os.environ['BACKEND'],
               reconfiguration_freq=1)

def snapshot():
    """Flat tensor of the full sim state."""
    d = env.unwrapped.get_state_dict()
    flat = []
    def walk(x):
        if isinstance(x, dict):
            for k in sorted(x): walk(x[k])
        else:
            flat.append(torch.as_tensor(x).reshape(1, -1).float().cpu())
    walk(d)
    return torch.cat(flat, dim=1)

def wrap(a):
    """(-pi, pi]. The one convention; see CLAUDE.md."""
    return -((-a + math.pi) % (2 * math.pi) - math.pi)

def cubes():
    u = env.unwrapped
    def yaw(q):
        w, x, y, z = [float(v) for v in q[0]]
        return wrap(math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))
    pa, pb = u.cubeA.pose.p[0], u.cubeB.pose.p[0]
    sep = math.dist([float(pa[0]), float(pa[1])], [float(pb[0]), float(pb[1])])
    return sep, wrap(yaw(u.cubeB.pose.q) - yaw(u.cubeA.pose.q))

# --- (a) same seed -> same initial state ---------------------------------
env.reset(seed=SEED); a = snapshot()
env.reset(seed=SEED); b = snapshot()
same = torch.allclose(a, b, atol=1e-6)
print(f'  reset(seed={SEED}) twice -> identical: {same}')
assert same, ('seed does not determine initial state. T-II cannot resample a '
              'failure region you cannot reproduce.')

# --- (b) distinct seeds -> distinct states -------------------------------
K = 32
states, seps = [], []
for i in range(K):
    env.reset(seed=SEED + i)
    states.append(snapshot())
    s, ry = cubes()
    seps.append(s)
uniq = torch.unique(torch.cat(states, dim=0), dim=0).shape[0]
print(f'  distinct initial states across {K} seeds: {uniq}')
assert uniq > K // 2, (
    f'only {uniq} distinct states from {K} seeds - seeds are not driving the '
    'initial-state randomisation, so your "100 rollouts" is really far fewer.')

# --- (c) the T-II failure axis is present and looks right ----------------
lo, hi = min(seps), max(seps)
print(f'  cube separation over {K} seeds: {lo*1000:.0f}-{hi*1000:.0f} mm '
      f'(sampler floor is 58.6 mm, cubes are 40 mm)')
assert lo > 0.0585, f'separation {lo*1000:.1f} mm is below the sampler floor'

# --- rollout timing ------------------------------------------------------
# Per-step cost on ONE cpu env. train.py evaluates with num_eval_envs
# subprocesses, so wall clock divides by that up to the core count.
env.reset(seed=SEED)
t = time.time()
STEPS = 100
for _ in range(STEPS):
    env.step(env.action_space.sample())
dt = time.time() - t
print(f'  {dt:.1f}s for {STEPS} steps = {dt/STEPS*1000:.1f} ms/step (1 cpu env)')
env.close()
PY
then ok; else bad "initial states are not reproducible, or the failure axis is missing"; fi

# =============================================================================
gate 5 "the eval path train.py actually uses, through evaluate()"
# =============================================================================
# physx_cpu vectorises with gym.vector.AsyncVectorEnv over a forkserver, one
# process per env. Different code path from gate 4's single env, and the one
# every eval during training runs through.
#
# This gate calls the REAL evaluate() with a stub agent rather than
# reimplementing it, because the failure it exists to catch lives inside
# evaluate(), not in env construction:
#
#   KeyError: 'final_info'   <- gymnasium >= 1.0. See CLAUDE.md traps.
#
# WRITTEN TO A FILE AND RUN, not piped in via heredoc like every other gate.
# multiprocessing's forkserver re-imports __main__ in each child, so __main__
# must be importable from a real path. Under `python - <<'PY'` it is not, the
# child dies during handshake, and the parent reports
#   ConnectionResetError: [Errno 104] Connection reset by peer
# which looks like a pod/network problem and is not one. Hence the file, the
# __main__ guard, and imports at module level so re-import restores sys.path.
DP_DIR=$MANISKILL_REPO/examples/baselines/diffusion_policy
G5=$OUT/gate5_eval.py
cat > "$G5" <<'PY'
import os, sys, time, torch, gymnasium
sys.path.insert(0, os.environ['DP_DIR'])
from diffusion_policy.make_env import make_eval_envs
from diffusion_policy.evaluate import evaluate

N = 2

def main():
    print(f'  gymnasium {gymnasium.__version__}')
    env_kwargs = dict(control_mode=os.environ['CTRL'], reward_mode='sparse',
                      obs_mode='state', render_mode='rgb_array',
                      max_episode_steps=5)
    t = time.time()
    envs = make_eval_envs(os.environ['ENV_ID'], N, os.environ['BACKEND'],
                          env_kwargs, dict(obs_horizon=2), video_dir=None)
    print(f'  built {type(envs).__name__}, {N} envs, in {time.time()-t:.1f}s')

    act_dim = envs.single_action_space.shape[0]

    class StubAgent:
        """Enough of the agent interface for evaluate(): zeros for an 8-step chunk."""
        def eval(self): pass
        def train(self): pass
        def get_action(self, obs): return torch.zeros((N, 8, act_dim))

    try:
        m = evaluate(N, StubAgent(), envs, 'cpu', os.environ['BACKEND'],
                     progress_bar=False)
    except KeyError as e:
        if 'final_info' in str(e):
            print('')
            print(f"  >> KeyError {e} with gymnasium {gymnasium.__version__}.")
            print('  >> gymnasium >= 1.0 removed final_info from vector envs and')
            print('  >> evaluate() reads it unguarded on the physx_cpu path.')
            print('  >> Fix: uv pip install --python $VENV/bin/python gymnasium==0.29.1')
        raise
    finally:
        envs.close()

    print('  evaluate() returned:', sorted(m.keys()))
    assert 'success_once' in m, \
        'evaluate() ran but reported no success_once - check record_metrics'
    print('  the full training eval path works end to end')

if __name__ == '__main__':
    main()
PY
if [ ! -d "$DP_DIR" ]; then
  skip "baseline dir missing: $DP_DIR"
elif DP_DIR="$DP_DIR" CTRL="$CTRL" ENV_ID="$ENV_ID" BACKEND="$BACKEND" \
     python "$G5"; then
  ok
else
  bad "async eval path broken - read the traceback above, do not assume a cause"
fi

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
echo "  All green. Next: bash t1/run_pipeline.sh data, then train."
echo ""
echo "  Not checked here: mp4 writing. --capture-video defaults to True, so"
echo "  the first training run proves it. T-II needs those videos - look at"
echo "  one rather than trusting that the file exists."
echo "=============================================================="
