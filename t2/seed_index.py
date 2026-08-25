#!/usr/bin/env python
"""Tabulate seed -> initial state, without running a policy.

    source /workspace/ripl/env.sh
    python t2/seed_index.py [n_seeds] [--start N] [--out seeds.csv]

reset(seed=s) fully determines the initial state: _initialize_episode runs under
torch.random.fork_rng() with manual_seed(_episode_seed), sapien_env.py:950-953.
So a seed is a lossless 8-byte encoding of the whole 70-float initial state, and
the map can be tabulated with resets alone - no policy, no 200-step rollout, no
GPU. Minutes for what would otherwise be hours.

This is what makes T-II's "resample FRESH episodes from the failure region"
cheap and, more importantly, correct. Filter this table for separation < 80 mm
and hand the surviving seeds to mine_rollouts.py: the episodes that come back are
drawn from exactly the environment's own conditional distribution given the
region. No state injection, no distribution shift, and every episode reproducible
from one integer.

The alternative - injecting cube poses via options={"reset_to_env_states": ...} -
replaces _initialize_episode wholesale, so the robot qpos and table pose would
have to be synthesised too, and under AsyncVectorEnv the options dict is
broadcast identically to every worker. Save injection for T-III, where a biased
distribution is the goal rather than a side effect.
"""
import csv
import os
import sys
import time

import gymnasium as gym
import numpy as np
import mani_skill.envs  # noqa: F401  (registers the envs)

from t2_common import cube_features, flatten_state_dict, manifest

N = int(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else 20000
START = int(sys.argv[sys.argv.index("--start") + 1]) if "--start" in sys.argv else 0
OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "seeds.csv"

ENV_ID = os.environ.get("ENV_ID", "StackCube-v1")
CTRL = os.environ.get("CTRL", "pd_ee_delta_pos")
BACKEND = os.environ.get("BACKEND", "physx_cpu")


def make(reconfig):
    return gym.make(ENV_ID, num_envs=1, obs_mode="state", control_mode=CTRL,
                    sim_backend=BACKEND, reconfiguration_freq=reconfig)


def poses(env):
    u = env.unwrapped
    return u.cubeA.pose.raw_pose, u.cubeB.pose.raw_pose


def check_reconfigure_is_irrelevant(k=12):
    """Index with reconfiguration_freq=0 only if it provably matches freq=1.

    make_eval_envs uses freq=1, so that is the setting the rollouts will run
    under. freq=0 is much faster because it skips rebuilding the scene every
    reset. _initialize_episode forks the RNG and reseeds from _episode_seed, so
    the two SHOULD agree - but "should" is how the PushT detour started, and
    this costs a couple of seconds to settle.
    """
    got = []
    for freq in (1, 0):
        env = make(freq)
        rows = []
        for s in range(START, START + k):
            env.reset(seed=s)
            a, b = poses(env)
            rows.append(np.concatenate([np.asarray(a).reshape(-1),
                                        np.asarray(b).reshape(-1)]))
        env.close()
        got.append(np.stack(rows))
    same = np.allclose(got[0], got[1], atol=1e-6)
    print(f"  reconfiguration_freq 0 vs 1 over {k} seeds: "
          f"{'identical' if same else 'DIFFER'}")
    if not same:
        d = np.abs(got[0] - got[1]).max()
        sys.exit(f"\n  Reconfiguring changes the sampled initial state (max diff "
                 f"{d:.2e}).\n  The index would then describe different episodes "
                 f"than the rollouts do.\n  Re-run the index with "
                 f"reconfiguration_freq=1 (slower) before trusting anything.\n")
    return same


def main():
    print(f"indexing seeds {START}..{START + N - 1} of {ENV_ID}")
    for k, v in manifest(n_seeds=N, start=START).items():
        print(f"  {k:14s} {v}")
    print("")

    fast = check_reconfigure_is_irrelevant()
    env = make(0 if fast else 1)

    f = open(OUT, "w", newline="")
    # Header is DERIVED from the first row, not declared. cube_features grew
    # the geometry fields (face_gap, dist_*) and a hardcoded fieldnames list
    # went stale silently until DictWriter raised 12,000 seeds into the run.
    # Every other writer in this harness builds its header from row.keys() for
    # exactly this reason; this one was the outlier.
    w = None

    state_rows, state_names, seps = [], None, []
    t0 = time.time()
    for i, s in enumerate(range(START, START + N)):
        env.reset(seed=s)
        a, b = poses(env)
        row = dict(seed=s, **cube_features(a, b))
        if w is None:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
        w.writerow(row)
        seps.append(row["separation"])

        names, vals = flatten_state_dict(env.unwrapped.get_state_dict())
        state_names = state_names or names
        state_rows.append(np.asarray(vals, dtype=np.float32))

        if i and i % 2000 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i + 1}/{N}  {rate:.0f} seeds/s  "
                  f"eta {(N - i - 1) / rate / 60:.1f} min", flush=True)
    f.close()
    env.close()

    # The initial state is redundant given the seed - it can always be replayed.
    # Saved anyway (~1.7 MB) so the analysis is reproducible from the committed
    # files alone, without a working ManiSkill install, and so the assumption
    # that seeds regenerate states is checkable rather than assumed. It also
    # carries the ONE part of the initial state the cube columns ignore: the
    # Panda's randomised initial qpos (robot_init_qpos_noise=0.02).
    npz = os.path.splitext(OUT)[0] + "_init_states.npz"
    np.savez_compressed(npz, seeds=np.arange(START, START + N),
                        states=np.stack(state_rows),
                        names=np.array(state_names))
    print(f"\nwrote {OUT} and {npz}  ({N} seeds, "
          f"{time.time() - t0:.0f}s, {len(state_names)} state dims)")

    # --- the sampler's own numbers, measured rather than simulated ----------
    seps = np.array(seps)
    print(f"\n  separation over {N} seeds (mm):")
    for q in (0, 1, 5, 10, 50, 100):
        print(f"    p{q:<3d} {np.percentile(seps, q) * 1000:7.1f}")
    for t in (0.060, 0.070, 0.080, 0.100):
        print(f"    P(sep < {t * 1000:.0f} mm) = {(seps < t).mean():.4f}   "
              f"-> {int((seps < t).sum())} of {N} seeds")

    # UniformPlacementSampler leaves unfilled rows at zeros when max_trials runs
    # out (samplers.py:52) and StackCube passes verbose=False for cubeB
    # (stack_cube.py:92), so a degenerate both-cubes-coincident state fails
    # silently. It would show up here as a separation near zero.
    floor = 0.0585
    bad = int((seps < floor).sum())
    if bad:
        print(f"\n  !! {bad} seeds below the sampler's {floor * 1000:.1f} mm floor "
              f"(min {seps.min() * 1000:.1f} mm).")
        print("     That is UniformPlacementSampler silently failing to place a "
              "cube. Drop those seeds.")
    else:
        print(f"\n  min separation {seps.min() * 1000:.1f} mm, clears the "
              f"{floor * 1000:.1f} mm sampler floor.")


if __name__ == "__main__":
    main()
