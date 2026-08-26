#!/usr/bin/env python
"""Capture the exact physx_cpu initial states of a finished evaluation pass.

The small piece the current t2/ miner no longer provides. It writes what
t4/backend_check.py replays, and it is cheap because it needs NO POLICY: reset
the env at each seed, read the sim state, move on. Minutes, not hours.

    python t4/capture_states.py --csv t2/results/mode_nominal_seed1.csv \\
                                t2/results/mode_nominal_seed2.csv \\
                                t2/results/mode_nominal_seed3.csv \\
        --out t4/results/nominal_states.npz

Row order is the CSVs' row order, concatenated in the order given, because
backend_check.py joins the two arms POSITIONALLY. It also stores the seed and
cubeA/cubeB coordinates it read back, so the join can be checked rather than
trusted.

Runs on physx_cpu, necessarily: the whole point is to capture what the CPU
backend actually produced for those seeds. On physx_cuda the same seeds give
different states, which is the fact this whole check exists to work around.
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "t2"))
from harness import CubePoseInfo, manifest, poses_from_info  # noqa: E402
from simstate import flatten_state_dict  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-envs", type=int, default=10)
    ap.add_argument("--max-episode-steps", type=int, default=200)
    a = ap.parse_args()

    backend = os.environ.get("BACKEND", "physx_cpu")
    if backend != "physx_cpu":
        sys.exit(f"\n!! BACKEND={backend}. These states must come from "
                 f"physx_cpu - they ARE the CPU arm.\n")

    rows = []
    for path in a.csv:
        with open(path) as f:
            for r in csv.DictReader(f):
                rows.append(dict(r, _src=os.path.basename(path)))
    seeds = [int(r["seed"]) for r in rows]
    print(f"capturing {len(seeds)} initial states from {len(a.csv)} block(s), "
          f"physx_cpu, no policy")

    sys.path.insert(0, f"{os.environ['MANISKILL_REPO']}/examples/baselines/"
                       f"diffusion_policy")
    from diffusion_policy.make_env import make_eval_envs
    from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

    env_id = os.environ.get("ENV_ID", "StackCube-v1")
    ctrl = os.environ.get("CTRL", "pd_ee_delta_pos")
    envs = make_eval_envs(
        env_id, a.num_envs, backend,
        dict(control_mode=ctrl, reward_mode="sparse", obs_mode="rgb",
             render_mode="rgb_array",
             human_render_camera_configs=dict(shader_pack="default"),
             max_episode_steps=a.max_episode_steps),
        dict(obs_horizon=2),
        wrappers=[FlattenRGBDObservationWrapper,
                  lambda e: CubePoseInfo(e, a.max_episode_steps)])

    names, flat, got_seeds, poses = None, [], [], []
    for start in range(0, len(seeds), a.num_envs):
        batch = seeds[start:start + a.num_envs]
        bseeds = batch + [batch[-1]] * (a.num_envs - len(batch))
        obs, info = envs.reset(seed=bseeds)
        n = len(batch)

        got = [int(np.asarray(v).reshape(-1)[0])
               for v in np.atleast_1d(info["episode_seed"])[:n]]
        for j, s in enumerate(batch):
            if got[j] != s:
                sys.exit(f"\n!! env {j} reset to seed {got[j]}, asked for {s}.\n")
        a0 = poses_from_info(info, "cubeA_pose", n)
        b0 = poses_from_info(info, "cubeB_pose", n)

        # BaseEnv.get_state_dict (sapien_env.py:1281), reached through
        # AsyncVectorEnv.call, which returns each worker's own return value
        # verbatim rather than aggregating it the way info is aggregated.
        # gym.Wrapper forwards __getattr__, so this passes through
        # CPUGymWrapper / FrameStack / CubePoseInfo untouched.
        try:
            sds = envs.call("get_state_dict")
        except (AttributeError, ValueError) as e:
            sys.exit(f"\n!! could not read the sim state through the vector "
                     f"env: {e}\n   Expected BaseEnv.get_state_dict to be "
                     f"reachable via envs.call().\n")
        for j in range(n):
            nm, vals = flatten_state_dict(sds[j])
            if names is None:
                names = nm
            elif nm != names:
                sys.exit("\n!! two envs produced different state layouts.\n")
            flat.append(vals)
            got_seeds.append(batch[j])
            poses.append([a0[j][0], a0[j][1], b0[j][0], b0[j][1]])
        print(f"  {min(start + a.num_envs, len(seeds)):>5}/{len(seeds)}", flush=True)

    envs.close()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    np.savez_compressed(
        a.out, names=np.asarray(names), init=np.asarray(flat, dtype=np.float32),
        seeds=np.asarray(got_seeds, dtype=np.int64),
        poses=np.asarray(poses, dtype=np.float64),
        src=np.asarray([r["_src"] for r in rows]))
    m = manifest(episodes=len(flat), state_width=len(names),
                 csvs=[os.path.abspath(p) for p in a.csv])
    import json
    with open(os.path.splitext(a.out)[0] + "_manifest.json", "w") as f:
        json.dump(m, f, indent=2)
    print(f"\nwrote {a.out}  ({len(flat)} states x {len(names)} floats)")


if __name__ == "__main__":
    main()
