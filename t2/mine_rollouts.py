#!/usr/bin/env python
"""Mine rollouts for T-II: many episodes, parallel, fully logged, resumable.

    source /workspace/ripl/env.sh
    python t2/mine_rollouts.py CKPT [--state] [--seeds A:B | --seed-file F]
                            [--repeats K] [--policy-seed N] [--num-envs N]
                            [--out PREFIX] [--max-hours H] [--trace-stride S]

Built for the overnight job rollout_log.py is not: rollout_log.py runs
num_envs=1 so it can reach env.unwrapped.cubeA in-process, which is right for a
50-episode localisation probe and roughly 10x too slow for the thousands of
episodes a failure characterisation needs. Here CubePoseInfo pushes the same
ground truth through info instead, so the run can go wide over AsyncVectorEnv.

WRITTEN TO A FILE AND RUN, never piped in as a heredoc: forkserver re-imports
__main__ in every child, and under `python - <<PY` it is not importable. The
child dies during handshake and the parent reports ConnectionResetError, which
reads like a pod fault and is not one. Hence the __main__ guard and module-level
imports. See CLAUDE.md's traps.

What comes out, per pass:
  PREFIX.csv              one row per episode - CLAUDE.md's schema plus the
                          taxonomy columns and cubeB_displacement
  PREFIX_trace.npz        every Sth step: cubeA/cubeB/tcp xyz + the 4 flags
  PREFIX_states.npz       full 70-float sim state at reset and at the last step
  PREFIX_manifest.json    ckpt, git shas, GPU, timings - what makes it a number

The trace is the part that separates locating a failure region from explaining
one. success_once tells you an episode failed; the trace tells you whether cubeB
was displaced during the approach or during the placement, and that is the
difference between two genuinely distinct failure modes.
"""
import csv
import json
import os
import sys
import time

import numpy as np
import torch

from t2_common import (build_agent, cube_features, flag, manifest, to_device)

MAX_EP_STEPS = int(os.environ.get("MAX_EP_STEPS", 200))
STEP_FLAGS = ("success", "is_cubeA_grasped", "is_cubeA_on_cubeB", "is_cubeA_static")


def arg(name, default=None, cast=str):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def seed_list():
    """--seeds A:B for a contiguous range, or --seed-file for a mined region.

    The seed file is how a targeted pass is driven: seed_index.py writes the
    seed -> initial state table, a filter picks the region, and the surviving
    seeds land here. One integer per line, or a CSV with a 'seed' column.
    """
    if "--seed-file" in sys.argv:
        path = arg("--seed-file")
        seeds = []
        with open(path) as f:
            head = f.readline()
            if "seed" in head and "," in head:          # a CSV from seed_index
                col = head.strip().split(",").index("seed")
                for line in f:
                    if line.strip():
                        seeds.append(int(line.split(",")[col]))
            else:
                for line in [head] + f.readlines():
                    if line.strip():
                        seeds.append(int(line.split(",")[0]))
        return seeds
    lo, hi = arg("--seeds", "0:100").split(":")
    return list(range(int(lo), int(hi)))


def main():
    ckpt = sys.argv[1]
    state_mode = "--state" in sys.argv
    repeats = arg("--repeats", 1, int)
    policy_seed = arg("--policy-seed", 1, int)
    num_envs = arg("--num-envs", 10, int)
    stride = arg("--trace-stride", 5, int)
    max_hours = arg("--max-hours", 1e9, float)
    seeds = seed_list()
    prefix = arg("--out") or (os.path.splitext(os.path.basename(ckpt))[0] + "_mined")

    # Each (seed, repeat) is one episode. Repeats exist because DDPM action
    # sampling is stochastic: a seed's outcome is a draw conditioned on the
    # initial state, not a property of it. Repeating separates "this state is
    # hard" from "the policy is noisy" - worth paying for in a targeted region,
    # not worth paying for over a nominal pass where distinct states are the
    # scarce resource.
    jobs = [(s, r) for r in range(repeats) for s in seeds]
    print(f"mining {len(jobs)} episodes = {len(seeds)} seeds x {repeats} repeats, "
          f"{num_envs} envs, ckpt {os.path.basename(ckpt)}")

    torch.manual_seed(policy_seed)
    np.random.seed(policy_seed)
    agent, envs, args, device = build_agent(ckpt, state_mode, num_envs,
                                            max_episode_steps=MAX_EP_STEPS)

    meta = manifest(ckpt=os.path.abspath(ckpt), obs_mode="state" if state_mode else "rgb",
                    n_seeds=len(seeds), repeats=repeats, policy_seed=policy_seed,
                    num_envs=num_envs, trace_stride=stride,
                    max_episode_steps=MAX_EP_STEPS, seed_min=min(seeds),
                    seed_max=max(seeds))
    for k, v in meta.items():
        print(f"  {k:16s} {v}")
    print("", flush=True)

    csv_f = open(prefix + ".csv", "w", newline="")
    csv_w = None
    traces, init_states, final_states, state_names = [], [], [], None
    t0 = time.time()
    done_eps = 0

    for start in range(0, len(jobs), num_envs):
        if (time.time() - t0) / 3600 > max_hours:
            print(f"\n  --max-hours {max_hours} reached; stopping cleanly at a "
                  f"batch boundary with {done_eps} episodes logged.")
            break

        batch = jobs[start:start + num_envs]
        # The last batch can be short. Pad it by repeating the final seed rather
        # than rebuilding the vector env at a smaller width, and drop the
        # padding rows afterwards.
        pad = num_envs - len(batch)
        bseeds = [s for s, _ in batch] + [batch[-1][0]] * pad

        obs, info = envs.reset(seed=bseeds)
        rows = [dict(run_id=prefix, ckpt_tag=os.path.basename(ckpt),
                     obs_mode="state" if state_mode else "rgb",
                     policy_seed=policy_seed, repeat_idx=r, seed=s, env_idx=j)
                for j, (s, r) in enumerate(batch)]

        a0 = np.stack([np.asarray(p, float).reshape(-1) for p in info["cubeA_pose"]])
        b0 = np.stack([np.asarray(p, float).reshape(-1) for p in info["cubeB_pose"]])
        for j, row in enumerate(rows):
            row.update(cube_features(a0[j], b0[j]))
            # episode_seed is read back from the env rather than assumed, so a
            # seed that silently failed to take shows up as a mismatch here
            # instead of as an unexplained outlier three plots later.
            got = int(np.asarray(info["episode_seed"][j]).reshape(-1)[0])
            if got != row["seed"]:
                sys.exit(f"\n  env {j} reset to seed {got}, expected {row['seed']}. "
                         f"The seed->state map does not hold; the index and the "
                         f"rollouts would describe different episodes.\n")
        init_states.append(np.stack([np.asarray(s) for s in info["env_state"]])[:len(batch)])
        if state_names is None:
            state_names = str(np.asarray(info["env_state_names"]).reshape(-1)[0]).split("|")

        # per-episode accumulators - this is what evaluate() throws away
        ever = {k: np.zeros(num_envs, bool) for k in STEP_FLAGS}
        first = {k: np.full(num_envs, -1) for k in STEP_FLAGS}
        trace, steps, fin = [], 0, [None] * num_envs

        done = False
        while not done and steps < MAX_EP_STEPS:
            with torch.no_grad():
                chunk = agent.get_action(to_device(obs, device))
            chunk = chunk.cpu().numpy()
            for i in range(chunk.shape[1]):
                obs, rew, term, trunc, info = envs.step(chunk[:, i])
                steps += 1

                # The vector env has already aggregated info into per-env
                # arrays, so read them positionally rather than per-dict.
                fl = {}
                for k in STEP_FLAGS:
                    v = np.asarray([bool(np.asarray(x).reshape(-1)[0])
                                    for x in np.atleast_1d(info[k])], bool) \
                        if k in info else np.zeros(num_envs, bool)
                    fl[k] = v
                    newly = v & ~ever[k]
                    first[k] = np.where(newly & (first[k] < 0), steps, first[k])
                    ever[k] = ever[k] | v

                if steps % stride == 0 or steps == MAX_EP_STEPS:
                    trace.append(np.concatenate([
                        np.stack([np.asarray(p, float).reshape(-1)[:3] for p in info["cubeA_pose"]]),
                        np.stack([np.asarray(p, float).reshape(-1)[:3] for p in info["cubeB_pose"]]),
                        np.stack([np.asarray(p, float).reshape(-1)[:3] for p in info["tcp_pose"]]),
                        np.stack([fl[k].astype(float) for k in STEP_FLAGS], axis=1),
                    ], axis=1).astype(np.float32))

                if trunc.any() or term.any():
                    done = True
                    # CPUGymWrapper(ignore_terminations=True, record_metrics=True)
                    # runs every episode to the full horizon and nests the real
                    # metrics under info['episode']. The top-level 'success' is
                    # the final-step value - success_at_end - NOT the
                    # success_once CLAUDE.md fixes as the reported number.
                    fin = list(np.atleast_1d(info.get("final_info", info)))
                    break

        for j, row in enumerate(rows):
            fi = fin[j] if isinstance(fin[j], dict) else {}
            epi = fi.get("episode", {}) if isinstance(fi, dict) else {}
            row["success_once"] = flag(epi, "success_once")
            row["success_at_end"] = flag(epi, "success_at_end")
            row["ep_len"] = int(np.asarray(epi.get("episode_len", steps)).reshape(-1)[0]) \
                if epi else steps
            row["return"] = float(np.asarray(epi.get("return", float("nan"))).reshape(-1)[0]) \
                if epi else float("nan")
            for k in STEP_FLAGS:
                row[k] = flag(fi, k)
                row["ever_" + k] = int(ever[k][j])
                row["first_" + k + "_step"] = int(first[k][j])
            af = np.asarray(fi.get("cubeA_pose", [np.nan] * 7), float).reshape(-1)
            bf = np.asarray(fi.get("cubeB_pose", [np.nan] * 7), float).reshape(-1)
            row.update(final_cubeA_x=af[0], final_cubeA_y=af[1], final_cubeA_z=af[2])
            # How far cubeB was shoved. The hypothesised close-separation
            # mechanism is that the descent onto A fouls B; this measures that
            # directly, turning the T-II claim from a correlation into a
            # mechanism for the price of one column.
            row["cubeB_displacement"] = float(np.hypot(bf[0] - b0[j][0], bf[1] - b0[j][1]))
            fs = fi.get("env_state")
            final_states.append(np.asarray(fs, np.float32) if fs is not None
                                else np.full(len(state_names or []), np.nan, np.float32))

        if csv_w is None:
            csv_w = csv.DictWriter(csv_f, fieldnames=list(rows[0].keys()))
            csv_w.writeheader()
        csv_w.writerows(rows)
        csv_f.flush()          # partial data beats an empty file on a dead job
        traces.append(np.stack(trace, axis=1)[:len(batch)])
        done_eps += len(batch)

        el = time.time() - t0
        rate = done_eps / el * 3600
        succ = np.mean([r["success_once"] for r in rows if r["success_once"] != ""])
        print(f"  {done_eps:>5}/{len(jobs)}  {rate:6.0f} ep/h  "
              f"eta {(len(jobs) - done_eps) / max(rate, 1e-9):5.2f} h  "
              f"batch success_once {succ:.2f}", flush=True)

    envs.close()
    csv_f.close()

    n = done_eps
    np.savez_compressed(prefix + "_trace.npz",
                        trace=np.concatenate(traces)[:n] if traces else np.zeros((0,)),
                        stride=stride,
                        columns=np.array(["cubeA_x", "cubeA_y", "cubeA_z",
                                          "cubeB_x", "cubeB_y", "cubeB_z",
                                          "tcp_x", "tcp_y", "tcp_z", *STEP_FLAGS]),
                        seeds=np.array([s for s, _ in jobs[:n]]),
                        repeats=np.array([r for _, r in jobs[:n]]))
    np.savez_compressed(prefix + "_states.npz",
                        init=np.concatenate(init_states)[:n] if init_states else np.zeros((0,)),
                        final=np.stack(final_states)[:n] if final_states else np.zeros((0,)),
                        names=np.array(state_names or []),
                        seeds=np.array([s for s, _ in jobs[:n]]))
    meta.update(episodes=n, wall_seconds=round(time.time() - t0, 1),
                episodes_per_hour=round(n / max(time.time() - t0, 1e-9) * 3600, 1))
    with open(prefix + "_manifest.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nwrote {prefix}.csv (+ _trace.npz, _states.npz, _manifest.json)  "
          f"{n} episodes in {(time.time() - t0) / 60:.1f} min "
          f"({meta['episodes_per_hour']:.0f} ep/h)")
    print("\n  Sanity-check that ep/h against a stopwatch before sizing the next")
    print("  run on it - CLAUDE.md's rule, and it was wrong by 10x on PushT.")


if __name__ == "__main__":
    main()
