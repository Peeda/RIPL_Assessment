#!/usr/bin/env python
"""Layer D: does the generated reward rank REAL episodes by their real outcome?

    python3 t3/align.py --run t3/artifacts/gap --mode gap --out $T3_OUT \
                        --ckpt $CKPT [--episodes 100] [--arms policy jitter zero]
                        [--reward t3/fixtures/stock_reward.py --label stock]

THE CENTREPIECE, AND WHY IT IS THE RIGHT TEST
    Everything else in this harness asks whether the reward is well-formed or
    whether a state we invented scores sensibly. This asks the only question
    that decides whether training on it is worth a GPU-day: run it as the
    environment's actual reward over a hundred real episodes of the frozen base
    policy, on the failure region's own initial states, and check that the
    episodes which succeeded accumulate more reward than the ones which failed.

    A reward that cannot separate observed success from observed failure on the
    policy we already have will not shape a residual toward success, whatever
    its rationale said.

THE CONDITIONAL TEST IS THE ONE THAT MATTERS
    An unconditional AUC is easy to score well on for the wrong reason. Grasping
    predicts nearly everything downstream, so a reward that pays lavishly for
    reaching and grasping - the commonest hack there is - separates outcomes
    handsomely while being useless for the failure we care about. So the gate
    also conditions on the stage the mode actually breaks at, from
    spec.MODE_STAGE:

        gap   fails at PLACEMENT      -> AUC( return ; ever_placed | grasped )
        farb  fails at HOLDING        -> AUC( return ; success | ever_placed )

    That is the number a grasp-farming reward lands at chance on.

THREE SEAMS, AND THE SECOND ONE IS A TRAP
    1. ENV_ID=StackCube-T3-v1 - t2/harness.py already reads it from the
       environment, so no code change. env_t3 is imported at MODULE scope
       because make_eval_envs calls gym.make inside a forkserver child.
    2. T3_SAMPLER=0, forced here rather than trusted. With the biased sampler
       on, reset(seed=s) no longer reproduces the T-II episode for s, and this
       measurement would be over a different population than the one whose
       outcomes it is being compared against. Recorded in the manifest so the
       gate can refuse a pass that got it wrong.
    3. reward_mode="dense" - the one line added to t2/harness.build_agent. The
       env then returns the generated reward and CPUGymWrapper(record_metrics)
       sums it into info['episode']['return'] (gymnasium.py:69) for nothing, so
       cumulative reward arrives beside success_once with no second rollout.

THE SEEDS ARE T-II's, NOT NEW ONES
    Taken from the `seed` column of mode_<tag>_seed1.csv - the fixed evaluation
    block T-IV is scored on. Reusing them means the alignment measurement and
    the eventual before/after are the same initial states, and it lets the gate
    join the logged poses back against T-II's own record: if StackCube-T3-v1
    perturbed the initial-state distribution in any way, that join fails.

CONTROL ARMS
    `jitter` wanders near the cube without doing the task - the "farm the
    shaping term" policy. `zero` does nothing. The generated reward must rank
    the real policy above both, by a ratio rather than a difference, so the
    threshold survives whatever scale the model chose.

WRITTEN TO A FILE AND RUN, never a heredoc: forkserver re-imports __main__ in
every child. See CLAUDE.md's traps.
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(os.path.dirname(HERE), "t2"))

import env_t3  # noqa: F401,E402  - registers StackCube-T3-v1 in every worker
from geometry import geom_features, yaw  # noqa: E402
from harness import build_agent, flag, manifest, poses_from_info, to_device  # noqa: E402
from spec import ALIGN_COLUMNS, REWARD_FILE  # noqa: E402

MAX_EP_STEPS = int(os.environ.get("MAX_EP_STEPS", 200))
STEP_FLAGS = ("is_cubeA_grasped", "is_cubeA_on_cubeB", "is_cubeA_static")
T2_RESULTS = os.environ.get("T2_RESULTS",
                            os.path.join(os.path.dirname(HERE), "t2", "results"))


def load_seeds(tag, n):
    """The fixed T-II evaluation block for this mode. -> (seeds, index rows)."""
    path = os.path.join(T2_RESULTS, f"mode_{tag}_seed1.csv")
    if not os.path.exists(path):
        sys.exit(
            f"\n!! {path} does not exist.\n\n"
            f"   Layer D measures the reward on the SAME episodes T-IV will be\n"
            f"   scored on, so it needs T-II's evaluation block. Run\n"
            f"     bash t2/run.sh eval\n"
            f"   and copy mode_{tag}_seed*.csv into {T2_RESULTS}, or point\n"
            f"   T2_RESULTS at the directory holding them.\n")
    with open(path) as f:
        rows = list(csv.DictReader(f))
    index = {int(r["seed"]): r for r in rows}
    return [int(r["seed"]) for r in rows][:n], index


def act(arm, agent, obs, device, n_act, num_envs, rng):
    """One action chunk for the named arm. -> (num_envs, horizon, n_act)."""
    if arm == "policy":
        with torch.no_grad():
            return agent.get_action(to_device(obs, device)).cpu().numpy()
    horizon = 8
    if arm == "zero":
        return np.zeros((num_envs, horizon, n_act), dtype=np.float32)
    # `jitter`: small translation noise, gripper held open. The policy that
    # loiters near the cubes without completing anything - which is what a
    # reward with an exploitable per-step shaping term pays for.
    a = rng.uniform(-0.3, 0.3, size=(num_envs, horizon, n_act)).astype(np.float32)
    a[:, :, -1] = 1.0
    return a


def run_arm(arm, agent, envs, device, tag, seeds, index, num_envs, policy_seed):
    rows = []
    rng = np.random.RandomState(policy_seed)
    n_act = envs.single_action_space.shape[-1]
    t0 = time.time()

    for start in range(0, len(seeds), num_envs):
        batch = seeds[start:start + num_envs]
        bseeds = batch + [batch[-1]] * (num_envs - len(batch))
        obs, info = envs.reset(seed=bseeds)
        n = len(batch)
        a0 = poses_from_info(info, "cubeA_pose", n)
        b0 = poses_from_info(info, "cubeB_pose", n)
        got = [int(np.asarray(v).reshape(-1)[0])
               for v in np.atleast_1d(info["episode_seed"])[:n]]

        batch_rows = []
        for j, s in enumerate(batch):
            # The env must have reset to the seed asked for. On physx_cpu this
            # assertion is meaningful; the T-II harness documents why it is not
            # on physx_cuda, which is why nothing here runs on GPU.
            if got[j] != s:
                sys.exit(f"\n!! asked for seed {s}, env reset to {got[j]}.\n")
            batch_rows.append(dict(
                run_id=f"align_{tag}_{arm}", mode=tag, arm=arm,
                policy_seed=policy_seed, seed=s,
                cubeA_x=a0[j][0], cubeA_y=a0[j][1],
                cubeB_x=b0[j][0], cubeB_y=b0[j][1]))

        ever = {k: np.zeros(num_envs, bool) for k in STEP_FLAGS}
        steps, fin = 0, [None] * num_envs
        done = False
        while not done and steps < MAX_EP_STEPS:
            chunk = act(arm, agent, obs, device, n_act, num_envs, rng)
            for i in range(chunk.shape[1]):
                obs, rew, term, trunc, info = envs.step(chunk[:, i])
                steps += 1
                for k in STEP_FLAGS:
                    if k in info:
                        ever[k] |= np.asarray(
                            [bool(np.asarray(x).reshape(-1)[0])
                             for x in np.atleast_1d(info[k])], bool)
                if trunc.any() or term.any():
                    done = True
                    fin = list(np.atleast_1d(info.get("final_info", info)))
                    break

        for j, row in enumerate(batch_rows):
            fi = fin[j] if isinstance(fin[j], dict) else {}
            epi = fi.get("episode", {}) if isinstance(fi, dict) else {}
            # 'return' is the sum of whatever reward_mode is active. Under
            # reward_mode="dense" on StackCube-T3-v1 that is the cumulative
            # GENERATED reward - the quantity this whole layer is about,
            # arriving for free from the wrapper T-II already uses.
            ret = epi.get("return", None)
            row["ep_return"] = ("" if ret is None
                                else float(np.asarray(ret).reshape(-1)[0]))
            row["ep_len"] = (int(np.asarray(epi.get("episode_len", steps))
                                 .reshape(-1)[0]) if epi else steps)
            row["ep_reward_mean"] = ("" if row["ep_return"] == ""
                                     else row["ep_return"] / max(row["ep_len"], 1))
            row["success_once"] = flag(epi, "success_once")
            row["success_at_end"] = flag(epi, "success_at_end")
            row["ever_grasped"] = int(ever["is_cubeA_grasped"][j])
            row["ever_placed"] = int(ever["is_cubeA_on_cubeB"][j])
            row["ever_static"] = int(ever["is_cubeA_static"][j])
            g = geom_features(row["cubeA_x"], row["cubeA_y"], row["cubeA_theta"],
                              row["cubeB_x"], row["cubeB_y"], row["cubeB_theta"])
            row.update({k: g[k] for k in
                        ("face_gap", "dist_A", "dist_B", "dist_max", "dist_min")})
            # Yaw computed from the pose the ENVIRONMENT produced, never
            # copied from the index row. The gate joins all six pose columns
            # against T-II's own record of these seeds, and a column copied
            # from that record would match trivially - proving nothing about
            # whether StackCube-T3-v1 reproduces the distribution.
            row["cubeA_theta"] = yaw(a0[j][3:7])
            row["cubeB_theta"] = yaw(b0[j][3:7])

        blank = [r["seed"] for r in batch_rows
                 if r["success_once"] == "" or r["ep_return"] == ""]
        if blank:
            sys.exit(
                f"\n!! {len(blank)} episode(s) came back with no metrics "
                f"(seeds {blank[:5]}...).\n"
                f"   CPUGymWrapper(record_metrics=True) nests 'return' and\n"
                f"   'success_once' under info['episode'], reached through\n"
                f"   info['final_info']. Their absence means the wrapper stack "
                f"is not\n   what this harness expects - NOT that the reward is "
                f"zero. Nothing written.\n")
        rows += batch_rows

        el = time.time() - t0
        got_r = [r["ep_return"] for r in rows]
        k = sum(1 for r in rows if r["success_once"] == 1)
        print(f"    {arm:<7} {len(rows):>4}/{len(seeds)}  "
              f"{len(rows) / el * 3600:6.0f} ep/h   "
              f"success {k / len(rows):.3f}   "
              f"mean return {sum(got_r) / len(got_r):9.2f}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--mode", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--reward", help="reward file to use instead of the "
                                     "generation's (the stock calibration arm)")
    ap.add_argument("--label", help="output label (default: mode)")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--control-episodes", type=int, default=20)
    ap.add_argument("--arms", nargs="+", default=["policy", "jitter", "zero"])
    ap.add_argument("--num-envs", type=int, default=10)
    ap.add_argument("--state", action="store_true")
    ap.add_argument("--policy-seed", type=int, default=1)
    a = ap.parse_args()

    # If a specific reward file was named, install it by pointing T3_RUN at a
    # directory containing it. env_t3 loads through the same path either way, so
    # the stock arm and the generated arm go through identical code - which is
    # the whole reason the stock arm is a useful comparison.
    if a.reward:
        import shutil
        shim = os.path.join(a.out, f"_shim_{a.label or 'alt'}")
        os.makedirs(shim, exist_ok=True)
        shutil.copyfile(a.reward, os.path.join(shim, REWARD_FILE))
        os.environ["T3_RUN"] = shim
    elif a.run:
        os.environ["T3_RUN"] = a.run

    # Not negotiable: with the biased sampler on, reset(seed=s) does not
    # reproduce the T-II episode for s and this measurement would be over the
    # wrong population. Forced here, and recorded in the manifest so the gate
    # can check it rather than take it on trust.
    os.environ["T3_SAMPLER"] = "0"
    os.environ["ENV_ID"] = "StackCube-T3-v1"

    label = a.label or a.mode
    os.makedirs(a.out, exist_ok=True)
    out_csv = os.path.join(a.out, f"align_{label}.csv")
    if os.path.exists(out_csv) and os.environ.get("FORCE", "0") != "1":
        with open(out_csv) as f:
            have = len(list(csv.DictReader(f)))
        want = (a.episodes + a.control_episodes * (len(a.arms) - 1))
        if have >= want:
            print(f"  {out_csv} already has {have} rows, skipping "
                  f"(FORCE=1 to redo)")
            return
        sys.exit(f"\n!! {out_csv} has {have} rows, expected {want}. That is a "
                 f"crashed run,\n   not a finished one. Delete it or set "
                 f"FORCE=1; do not append to it.\n")

    seeds, index = load_seeds(a.mode, a.episodes)
    print(f"  {env_t3.describe()}")
    print(f"  seeds       {len(seeds)} from mode_{a.mode}_seed1.csv "
          f"[{min(seeds)}..{max(seeds)}]")

    agent, envs, args, device = build_agent(
        a.ckpt, a.state, a.num_envs, max_episode_steps=MAX_EP_STEPS,
        reward_mode="dense")

    rows = []
    t0 = time.time()
    try:
        for arm in a.arms:
            n = a.episodes if arm == "policy" else a.control_episodes
            torch.manual_seed(a.policy_seed)
            np.random.seed(a.policy_seed)
            rows += run_arm(arm, agent, envs, device, a.mode, seeds[:n], index,
                            a.num_envs, a.policy_seed)
    finally:
        envs.close()

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ALIGN_COLUMNS)
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in ALIGN_COLUMNS} for r in rows])

    ns_path = a.reward or os.path.join(a.run, REWARD_FILE)
    from loader import load_file
    from spec import REWARD_MAX_NAME
    reward_max = float(load_file(ns_path, "reward")[REWARD_MAX_NAME])

    with open(os.path.join(a.out, f"align_{label}_manifest.json"), "w") as f:
        json.dump(manifest(ckpt=a.ckpt, mode=a.mode, label=label,
                           reward=os.path.relpath(ns_path),
                           reward_max=reward_max,
                           t3_sampler=int(env_t3.sampler_enabled()),
                           arms=a.arms, episodes=a.episodes,
                           control_episodes=a.control_episodes,
                           policy_seed=a.policy_seed,
                           seed_min=min(seeds), seed_max=max(seeds),
                           wall_seconds=round(time.time() - t0, 1)), f, indent=2)

    print(f"\n  wrote       {out_csv} ({len(rows)} episodes) and its manifest")
    print(f"  These are measurements. t3/verify.py computes the AUCs and "
          f"applies the thresholds.")


if __name__ == "__main__":
    main()
