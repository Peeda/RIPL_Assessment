#!/usr/bin/env python
"""Tabulate seed -> initial state, without running a policy.

    source /workspace/ripl/env.sh
    python t2/seed_index.py [n_seeds] [--start N] [--out seeds.csv]

reset(seed=s) fully determines the initial state: _initialize_episode runs under
torch.random.fork_rng() with manual_seed(_episode_seed), sapien_env.py:950-953.
So a seed is a lossless 8-byte encoding of the whole initial state, and the map
can be tabulated with resets alone - no policy, no 200-step rollout, no GPU.
Minutes for what would otherwise be hours.

THIS IS THE LOAD-BEARING IDEA OF THE WHOLE HARNESS. It turns T-II's "resample
FRESH episodes from the failure region" into rejection sampling over integers:
filter this table for the region, hand the surviving seeds to eval_modes.py. The
episodes that come back are drawn from exactly the environment's own conditional
distribution given the region. No state injection, no distribution shift, and
every episode reproducible from one integer.

The alternative - injecting cube poses via options={"reset_to_env_states": ...} -
replaces _initialize_episode wholesale, so the robot qpos and table pose would
have to be synthesised too, and under AsyncVectorEnv the options dict is
broadcast identically to every worker. Save injection for T-III, where a biased
distribution is the goal rather than a side effect.

The index is also HALF OF THE VERIFICATION. verify.py joins the cube poses
logged during a rollout against the poses recorded here; this script and
eval_modes.py reach the simulator by different paths, so their agreeing means
the seed -> state map is genuine and deterministic, and that the rollouts really
did run on the episodes the filter selected.
"""
import csv
import os
import sys
import time

import gymnasium as gym
import numpy as np
import mani_skill.envs  # noqa: F401  (registers the envs)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry import EVAL_BASE, MODES, cube_features  # noqa: E402
from harness import manifest  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else 25000
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
    this costs a couple of seconds to settle. If they ever disagreed, the index
    would describe different episodes than the rollouts do, and verify.py's
    index join would start failing for a reason nobody would guess.
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
    # Header DERIVED from the first row, not declared. cube_features has grown
    # columns twice, and a hardcoded fieldnames list went stale silently until
    # DictWriter raised 12,000 seeds into a run.
    w = None
    rows, seps = [], []
    t0 = time.time()
    for i, s in enumerate(range(START, START + N)):
        env.reset(seed=s)
        a, b = poses(env)
        row = dict(seed=s, **cube_features(np.asarray(a).reshape(-1),
                                           np.asarray(b).reshape(-1)))
        if w is None:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
        w.writerow(row)
        rows.append(row)
        seps.append(row["separation"])

        if i and i % 2000 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i + 1}/{N}  {rate:.0f} seeds/s  "
                  f"eta {(N - i - 1) / rate / 60:.1f} min", flush=True)
    f.close()
    env.close()
    print(f"\nwrote {OUT}  ({N} seeds, {time.time() - t0:.0f}s)")

    # --- the sampler's own numbers, measured rather than assumed ------------
    seps = np.array(seps)
    print(f"\n  separation over {N} seeds (mm):")
    for q in (0, 1, 5, 10, 50, 100):
        print(f"    p{q:<3d} {np.percentile(seps, q) * 1000:7.1f}")

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

    # --- can this index actually fill an evaluation? ------------------------
    #
    # Free, and it is what stops a pass discovering at hour three that a region
    # is too rare to fill. eval_modes.py draws only from seeds >= EVAL_BASE, so
    # that is the pool this reports on.
    pool = [r for r in rows if r["seed"] >= EVAL_BASE]
    print(f"\n  region availability above seed {EVAL_BASE} "
          f"({len(pool)} eligible seeds):")
    for tag, pred in MODES.items():
        hits = sum(1 for r in pool if pred(r))
        rate = hits / max(len(pool), 1)
        need = EVAL_BASE + int(300 / rate) if rate else 0
        flag = "ok" if hits >= 300 else "!! SHORT of the 300 a 3 x 100 pass needs"
        print(f"    {tag:9s} {hits:6d}  ({rate:6.2%})  "
              f"300 seeds need an index of ~{need:,}   {flag}")


if __name__ == "__main__":
    main()
