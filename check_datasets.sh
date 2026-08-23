#!/usr/bin/env bash
# ==============================================================================
# What is actually on disk, and is it usable?
#
#   source /workspace/ripl/env.sh && bash check_datasets.sh
#
# Answers three questions the filesystem alone cannot:
#   1. Is each replay output complete, or was it produced with the old flag?
#   2. Does its action dim match the control mode you intend to train under?
#   3. Which source file did it come from?
#
# Read-only. Touches nothing.
# ==============================================================================
set -uo pipefail

: "${MS_ASSET_DIR:?source /workspace/ripl/env.sh first}"
DEMOS=$MS_ASSET_DIR/demos/PushT-v1/rl
CTRL=${CTRL:-pd_ee_delta_pos}
MIN_TRAJS=${MIN_TRAJS:-600}

echo "=============================================================="
echo "PushT-v1 demo directory: $DEMOS"
echo "intended control mode:   $CTRL"
echo "=============================================================="
ls -1 "$DEMOS"/*.h5 2>/dev/null | sed 's|.*/|  |' || { echo "  (none)"; exit 1; }
echo ""

CTRL="$CTRL" MIN_TRAJS="$MIN_TRAJS" DEMOS="$DEMOS" python - <<'PY'
import os, glob, json, h5py

demos = os.environ['DEMOS']
ctrl = os.environ['CTRL']
minimum = int(os.environ['MIN_TRAJS'])

# pd_ee_delta_pos = 3-DoF translation; pd_ee_delta_pose = 6-DoF with rotation.
# PushT appends a gripper/extra dim in some configs, so treat these as hints
# and let the observed value speak — the check that matters is that the demo
# dim equals what gym.make reports for the SAME control mode.
problems = []

# Index the shipped obs_mode=none sources by control mode. Episode counts
# differ per control mode (PushT ships 888 / 719 / 999 for pos / pose /
# joint), so a single hardcoded "expected" number is wrong for at least two
# of them.
sources = {}
for p in glob.glob(os.path.join(demos, 'trajectory.none.*.h5')):
    parts = os.path.basename(p).split('.')
    if len(parts) > 3:
        try:
            with h5py.File(p, 'r') as f:
                sources[parts[2]] = len(f.keys())
        except Exception:
            pass
if sources:
    print("  shipped sources (episodes available per control mode):")
    for k, v in sorted(sources.items()):
        mark = '  <- intended' if k == ctrl else ''
        print(f"    {k:<20} {v}{mark}")
    print("")

expected = sources.get(ctrl, 0)
minimum = int(expected * 0.85) if expected else minimum
if expected:
    print(f"  completeness threshold for {ctrl}: {minimum} "
          f"(85% of {expected})\n")

for path in sorted(glob.glob(os.path.join(demos, '*.h5'))):
    name = os.path.basename(path)
    parts = name.split('.')
    obs_mode = parts[1] if len(parts) > 2 else '?'
    ctrl_mode = parts[2] if len(parts) > 3 else '?'

    try:
        with h5py.File(path, 'r') as f:
            keys = list(f.keys())
            n = len(keys)
            adim = f[keys[0]]['actions'].shape[-1] if n else None
            has_obs = 'obs' in f[keys[0]] if n else False
    except Exception as e:
        print(f"  {name}\n      UNREADABLE: {e}\n")
        problems.append(name)
        continue

    size_gb = os.path.getsize(path) / 1e9
    print(f"  {name}")
    print(f"      obs_mode={obs_mode}  control_mode={ctrl_mode}  "
          f"{n} trajectories  {size_gb:.2f} GB  action_dim={adim}")

    side = path[:-3] + '.json'
    n_own = 0
    if os.path.exists(side):
        try:
            d = json.load(open(side))
            n_own = len(d.get('episodes', []))
            print(f"      source_type={d.get('source_type')}  "
                  f"episodes_in_own_metadata={n_own}")
        except Exception:
            pass

    # Yield must be measured against the SOURCE file's episode count, not this
    # file's own sidecar. replay_trajectory writes a fresh sidecar listing only
    # the episodes it KEPT, so n/n_own is 100% by construction and tells you
    # nothing. The source is the trajectory.none.<same control mode>.*.h5 file.
    if obs_mode != 'none':
        src_n = sources.get(ctrl_mode, 0)
        if src_n:
            pct = 100.0 * n / src_n
            print(f"      yield vs source ({ctrl_mode}): {n}/{src_n} = {pct:.1f}%")
            if n_own == n and n < src_n * 0.8:
                print(f"      note: own metadata lists {n_own}, matching the "
                      f"trajectory count exactly. That is the signature of a "
                      f"--count-capped run, not a low-yield one.")
        else:
            print(f"      (no source file for control mode {ctrl_mode}; "
                  f"cannot compute yield)")

    # obs_mode=none files are the shipped sources — they are not replay outputs
    # and the count check does not apply to them.
    if obs_mode != 'none':
        own_src = sources.get(ctrl_mode, 0)
        thresh = int(own_src * 0.85) if own_src else minimum
        if n < thresh:
            print(f"      >> SHORT: {n} < {thresh}. Either a --count-capped "
                  f"run or a low-yield replay. Regenerate at full scale with "
                  f"--use-env-states.")
            problems.append(name)
        if ctrl_mode != ctrl:
            print(f"      >> CONTROL MODE MISMATCH: built for {ctrl_mode}, "
                  f"you intend to train under {ctrl}.")
            problems.append(name)
        if not has_obs:
            print(f"      >> no 'obs' group — replay did not attach "
                  f"observations.")
            problems.append(name)
    print("")

print("=" * 62)
if problems:
    print("Needs attention:")
    for p in sorted(set(problems)):
        print(f"  - {p}")
else:
    print("All replay outputs look complete and consistent.")
print("=" * 62)
PY

echo ""
echo "action-dim cross-check against a live env under $CTRL:"
CTRL="$CTRL" python - <<'PY' 2>&1 | sed 's/^/  /'
import os, gymnasium as gym, mani_skill.envs
ctrl = os.environ['CTRL']
env = gym.make('PushT-v1', num_envs=1, obs_mode='state',
               control_mode=ctrl, sim_backend='physx_cuda')
print(f"gym.make(control_mode={ctrl!r}) -> action_space {env.action_space.shape}")
env.close()
PY

cat <<'EOF'

--------------------------------------------------------------------------
If the state dataset above reports control_mode=pd_ee_delta_pose while you
intend to train under pd_ee_delta_pos, it is not reusable for T-I. The action
spaces differ in dimension, so it will fail the gate-9 check rather than
silently mistrain — but it does mean the 694 trajectories must be replayed
again under the new control mode. That pass is cheap (minutes at 1024 envs);
the RGB pass is the expensive one, so settle the control mode before starting
it, not after.

The old dataset is still worth keeping for the throughput sweeps — worker
counts and batch sizes transfer across control modes.
--------------------------------------------------------------------------
EOF
