#!/usr/bin/env python
"""Measure the generated artifacts. Three stages, one file. Needs ManiSkill.

    python3 t3/check.py sampler --run DIR --mode gap --out OUT [--draws 4096]
    python3 t3/check.py reward  --run DIR --out OUT [--reward FILE --label stock]
    python3 t3/check.py align   --run DIR --mode gap --out OUT --ckpt CKPT
                                [--episodes 100] [--reward FILE --label stock]

EVERY STAGE MEASURES AND WRITES A FILE. NONE OF THEM DECIDES.
    t3/summary.py reads what these write, prints OK/WARN, and exits 0. The split
    is what lets the thresholds be reviewed, and argued with, on a laptop with
    no GPU - and it is why a disappointing number is a finding in the report
    rather than a command that failed.

WRITTEN TO A FILE AND RUN, never a heredoc, and `import env_t3` is at MODULE
scope. physx_cpu vectorises by subprocess; forkserver re-imports __main__ in
every child, so a register_env inside a function does not exist in the workers
and gym.make dies there on an unknown env id. See CLAUDE.md's traps. That same
constraint is why t3/summary.py is a separate file - it has to run on a laptop
with no ManiSkill, and this one cannot.
"""
import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(os.path.dirname(HERE), "t2"))

import gymnasium as gym  # noqa: E402
import mani_skill.envs  # noqa: F401,E402
import env_t3  # noqa: F401,E402  - registers StackCube-T3-v1 in every worker
from geometry import MODES, geom_features, geom_from_row, yaw  # noqa: E402
from harness import build_agent, flag, manifest, poses_from_info, to_device  # noqa: E402
from loader import load_file  # noqa: E402
from spec import (ALIGN_COLUMNS, CUBE_Z, MIN_SEPARATION, REACH_MAX,  # noqa: E402
                  REWARD_FILE, REWARD_MAX_NAME, SAMPLER_COLUMNS, SAMPLER_FILE,
                  SUPPORT_X, SUPPORT_Y, mean, sd)

MAX_EP_STEPS = int(os.environ.get("MAX_EP_STEPS", 200))
STEP_FLAGS = ("is_cubeA_grasped", "is_cubeA_on_cubeB", "is_cubeA_static")
T2_RESULTS = os.environ.get("T2_RESULTS",
                            os.path.join(os.path.dirname(HERE), "t2", "results"))


# ===========================================================================
# sampler
# ===========================================================================


def base_rate(tag):
    """The nominal probability of the region, from the committed seed index.

    Computed at run time rather than hardcoded: the index is policy-free resets
    of the stock environment, so this is the environment's own rate for the
    region - the number the sampler's hit rate has to be an enrichment over.
    """
    path = os.path.join(T2_RESULTS, "seeds.csv")
    if not os.path.exists(path):
        return None, 0
    n = hit = 0
    with open(path) as f:
        for r in csv.DictReader(f):
            n += 1
            hit += bool(MODES[tag](geom_from_row(r)))
    return (hit / n if n else None), n


def draw(run, n, device, seed):
    torch.manual_seed(seed)
    ns = load_file(os.path.join(run, SAMPLER_FILE), "sampler")
    with torch.device(device):
        return ns["sample_cube_poses"](n, device)


def sampler_rows(out):
    """The sampler's tensors -> plain dicts, via t2/geometry.py."""
    got = {}
    for key, w in (("cubeA_xyz", 3), ("cubeA_quat", 4),
                   ("cubeB_xyz", 3), ("cubeB_quat", 4)):
        t = out.get(key)
        if t is None:
            sys.exit(f"\n!! the sampler returned no '{key}' (got {sorted(out)}).\n")
        if t.dim() != 2 or t.shape[1] != w:
            sys.exit(f"\n!! the sampler's '{key}' has shape {tuple(t.shape)}, "
                     f"expected (b, {w}).\n")
        if not torch.isfinite(t).all():
            sys.exit(f"\n!! the sampler's '{key}' contains non-finite values.\n")
        got[key] = t.detach().cpu().tolist()

    rows = []
    for i in range(len(got["cubeA_xyz"])):
        a, b = got["cubeA_xyz"][i], got["cubeB_xyz"][i]
        ta, tb = yaw(got["cubeA_quat"][i]), yaw(got["cubeB_quat"][i])
        rows.append(dict(
            i=i, cubeA_x=a[0], cubeA_y=a[1], cubeA_theta=ta,
            cubeB_x=b[0], cubeB_y=b[1], cubeB_theta=tb,
            separation=math.dist((a[0], a[1]), (b[0], b[1])),
            _az=a[2], _bz=b[2],
            _aq=got["cubeA_quat"][i], _bq=got["cubeB_quat"][i],
            **geom_features(a[0], a[1], ta, b[0], b[1], tb)))
    return rows


def stage_sampler(a):
    print(f"  {env_t3.describe()}")
    print(f"  sampler     {a.run}/{SAMPLER_FILE}, mode '{a.mode}', "
          f"{a.draws} draws")
    rows = sampler_rows(draw(a.run, a.draws, "cpu", 0))

    # The one property that catches a sampler drawing from `random` or numpy,
    # which every other check here would pass: reset(seed=s) seeds torch's
    # generator and nothing else, so a non-torch draw is not reproducible.
    d1, d2 = draw(a.run, 256, "cpu", 12345), draw(a.run, 256, "cpu", 12345)
    deterministic = all(torch.equal(d1[k], d2[k]) for k in d1)

    n = len(rows)
    br, br_n = base_rate(a.mode)
    hits = sum(1 for r in rows if MODES[a.mode](r))
    m = dict(
        mode=a.mode, draws=n,
        quat_bad=sum(1 for r in rows for q in (r["_aq"], r["_bq"])
                     if abs(math.sqrt(sum(v * v for v in q)) - 1) > 1e-5
                     or abs(q[1]) > 1e-6 or abs(q[2]) > 1e-6),
        z_bad=sum(1 for r in rows for z in (r["_az"], r["_bz"])
                  if abs(z - CUBE_Z) > 1e-6),
        separation_bad=sum(1 for r in rows
                           if r["separation"] < MIN_SEPARATION - 1e-9),
        support_bad=sum(
            1 for r in rows
            if not (SUPPORT_X[0] - 1e-6 <= r["cubeA_x"] <= SUPPORT_X[1] + 1e-6
                    and SUPPORT_X[0] - 1e-6 <= r["cubeB_x"] <= SUPPORT_X[1] + 1e-6
                    and SUPPORT_Y[0] - 1e-6 <= r["cubeA_y"] <= SUPPORT_Y[1] + 1e-6
                    and SUPPORT_Y[0] - 1e-6 <= r["cubeB_y"] <= SUPPORT_Y[1] + 1e-6)),
        reach_bad=sum(1 for r in rows
                      if r["dist_A"] > REACH_MAX or r["dist_B"] > REACH_MAX),
        hits=hits, hit_rate=hits / n,
        base_rate=br, base_rate_n=br_n,
        enrichment=(hits / n / br) if br else None,
        sd_cubeA_x=sd(r["cubeA_x"] for r in rows),
        sd_cubeA_y=sd(r["cubeA_y"] for r in rows),
        sd_cubeB_x=sd(r["cubeB_x"] for r in rows),
        sd_cubeB_y=sd(r["cubeB_y"] for r in rows),
        sd_face_gap=sd(r["face_gap"] for r in rows),
        mean_face_gap=mean(r["face_gap"] for r in rows),
        mean_dist_A=mean(r["dist_A"] for r in rows),
        mean_dist_B=mean(r["dist_B"] for r in rows),
        deterministic=deterministic)

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"sampler_{a.mode}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SAMPLER_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (int(MODES[a.mode](r)) if k == "in_region" else r[k])
                        for k in SAMPLER_COLUMNS})
    with open(os.path.join(a.out, f"sampler_{a.mode}.json"), "w") as f:
        json.dump(m, f, indent=2)

    enr = f"{m['enrichment']:.1f}x" if m["enrichment"] else "?"
    print(f"  hit rate    {m['hit_rate']:.3f}  (nominal "
          f"{m['base_rate']:.4f} over {m['base_rate_n']} indexed seeds -> {enr})"
          if m["base_rate"] else f"  hit rate    {m['hit_rate']:.3f}")
    print(f"  invalid     quat {m['quat_bad']}  z {m['z_bad']}  "
          f"separation {m['separation_bad']}  support {m['support_bad']}  "
          f"reach {m['reach_bad']}")
    print(f"  spread      sd(A) {m['sd_cubeA_x']:.3f}/{m['sd_cubeA_y']:.3f}  "
          f"sd(gap) {m['sd_face_gap']:.4f}   deterministic {deterministic}")


# ===========================================================================
# reward - shape, finiteness, bounds, purity, no mutation
# ===========================================================================
#
# Cheap and unglamorous, and it catches the errors that would otherwise surface
# as a training run that diverges for no visible reason: a reward that returns a
# Python float and broadcasts, one that returns (n, 1) and turns the advantage
# into a matrix, one that divides by a distance that is zero exactly when the
# gripper reaches the cube.
#
# Two of these are worth more than they look. PURITY: a single torch.rand_like
# buried in a shaping term makes every advantage estimate noisy in a way no
# amount of training averages out, and nothing else here would notice. NO
# MUTATION: the nastiest hack available is a reward that moves cubeA onto cubeB
# itself and then collects the bonus - the static check bans .set_pose by name,
# this does not depend on having thought of the right method name.


def stage_reward(a):
    path = a.reward or os.path.join(a.run, REWARD_FILE)
    ns = load_file(path, "reward")
    fn, reward_max = ns["compute_reward"], float(ns[REWARD_MAX_NAME])
    label = a.label or a.mode
    print(f"  {env_t3.describe()}")
    print(f"  reward      {path}  ({REWARD_MAX_NAME}={reward_max})")

    os.environ["T3_SAMPLER"] = "0"
    env = gym.make("StackCube-T3-v1", num_envs=1, obs_mode="state",
                   sim_backend="physx_cpu", reconfiguration_freq=0,
                   reward_mode="sparse",
                   control_mode=os.environ.get("CTRL", "pd_ee_delta_pos"),
                   max_episode_steps=MAX_EP_STEPS)
    u = env.unwrapped
    res = dict(reward=path, reward_max=reward_max, label=label, n_calls=0,
               shape_ok=True, dtype_ok=True, finite_ok=True, device_ok=True,
               pure_ok=True, mutation_ok=True,
               r_min=float("inf"), r_max=float("-inf"), note="")
    try:
        for s in (10001, 10002):
            obs, _ = env.reset(seed=s)
            for _ in range(20):
                info = u.evaluate()
                action = env.action_space.sample()
                at = torch.as_tensor(np.asarray(action)).float().reshape(1, -1)

                r = fn(u, obs, at, info)
                res["n_calls"] += 1
                if not torch.is_tensor(r):
                    res["shape_ok"] = res["dtype_ok"] = False
                    res["note"] = f"returned {type(r).__name__}, not a tensor"
                    break
                if tuple(r.shape) != (u.num_envs,):
                    res["shape_ok"] = False
                    res["note"] = f"shape {tuple(r.shape)} != ({u.num_envs},)"
                res["dtype_ok"] &= bool(torch.is_floating_point(r))
                res["device_ok"] &= str(r.device) == str(u.device)
                res["finite_ok"] &= bool(torch.isfinite(r).all())
                v = r.detach().cpu().float().reshape(-1)
                res["r_min"] = min(res["r_min"], float(v.min()))
                res["r_max"] = max(res["r_max"], float(v.max()))

                res["pure_ok"] &= bool(torch.equal(r.detach(),
                                                   fn(u, obs, at, info).detach()))
                before = u.get_state().clone()
                fn(u, obs, at, info)
                res["mutation_ok"] &= bool(
                    torch.allclose(before, u.get_state(), atol=0, rtol=0))

                obs, _, _, _, _ = env.step(action)
    finally:
        env.close()

    res["bounds_ok"] = (res["r_min"] >= -1e-5
                        and res["r_max"] <= reward_max + 1e-5)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"reward_{label}.json"), "w") as f:
        json.dump(res, f, indent=2)

    print(f"  {res['n_calls']} calls   shape {res['shape_ok']}  "
          f"finite {res['finite_ok']}  pure {res['pure_ok']}  "
          f"no-mutation {res['mutation_ok']}  "
          f"range [{res['r_min']:.3f}, {res['r_max']:.3f}]")

    # Shape and finiteness are the two that would crash or silently corrupt PPO,
    # so they stop the pipeline here. Bounds and purity are warnings that
    # summary.py reports.
    if not (res["shape_ok"] and res["finite_ok"]):
        sys.exit(f"\n!! the reward is not usable: {res['note'] or 'see above'}\n"
                 f"   A wrong shape broadcasts into the advantage and a "
                 f"non-finite value poisons\n   the whole batch. Regenerate; do "
                 f"not train on this.\n")


# ===========================================================================
# align - does the reward rank REAL episodes by their real outcome?
# ===========================================================================
#
# The centrepiece, and the only test that decides whether training on this is
# worth a GPU-day: run the generated reward as the environment's actual reward
# over 100 real episodes of the frozen base policy, on the failure region's own
# initial states, and check the episodes that succeeded accumulated more reward
# than the ones that failed.
#
# THE SEEDS ARE T-II's, NOT NEW ONES. Taken from the `seed` column of
# mode_<tag>_seed1.csv - the fixed evaluation block T-IV is scored on - so this
# measurement and the eventual before/after are the same initial states.
#
# THREE SEAMS: ENV_ID=StackCube-T3-v1 (t2/harness.py reads it from the
# environment, so no code change); T3_SAMPLER=0, forced here rather than
# trusted, because with the biased sampler on reset(seed=s) no longer reproduces
# the T-II episode for s and this would be over the wrong population;
# reward_mode="dense", the one line added to t2/harness.build_agent, after which
# CPUGymWrapper(record_metrics) sums the generated reward into
# info['episode']['return'] for free.


def load_seeds(tag, n):
    path = os.path.join(T2_RESULTS, f"mode_{tag}_seed1.csv")
    if not os.path.exists(path):
        sys.exit(f"\n!! {path} does not exist.\n\n"
                 f"   This measures the reward on the SAME episodes T-IV will "
                 f"be scored on,\n   so it needs T-II's evaluation block. Run "
                 f"`bash t2/run.sh eval`, or point\n   T2_RESULTS at the "
                 f"directory holding mode_{tag}_seed*.csv.\n")
    with open(path) as f:
        return [int(r["seed"]) for r in csv.DictReader(f)][:n]


def act(arm, agent, obs, device, n_act, num_envs, rng):
    if arm == "policy":
        with torch.no_grad():
            return agent.get_action(to_device(obs, device)).cpu().numpy()
    horizon = 8
    if arm == "zero":
        return np.zeros((num_envs, horizon, n_act), dtype=np.float32)
    # `jitter`: the policy that loiters near the cubes without completing
    # anything, which is what an exploitable per-step shaping term pays for.
    a = rng.uniform(-0.3, 0.3, size=(num_envs, horizon, n_act)).astype(np.float32)
    a[:, :, -1] = 1.0
    return a


def run_arm(arm, agent, envs, device, tag, seeds, num_envs, policy_seed):
    rows = []
    rng = np.random.RandomState(policy_seed)
    n_act = envs.single_action_space.shape[-1]
    t0 = time.time()

    for start in range(0, len(seeds), num_envs):
        batch = seeds[start:start + num_envs]
        bseeds = batch + [batch[-1]] * (num_envs - len(batch))
        if hasattr(agent, "reset_chunk"):
            agent.reset_chunk()  # T-IV's ResidualAgent; a no-op with no residual
        obs, info = envs.reset(seed=bseeds)
        n = len(batch)
        a0 = poses_from_info(info, "cubeA_pose", n)
        b0 = poses_from_info(info, "cubeB_pose", n)
        got = [int(np.asarray(v).reshape(-1)[0])
               for v in np.atleast_1d(info["episode_seed"])[:n]]

        batch_rows = []
        for j, s in enumerate(batch):
            # The env must have reset to the seed asked for. On physx_cpu this
            # assertion is meaningful; T-II documents why it is not on
            # physx_cuda, which is why nothing here runs on GPU.
            if got[j] != s:
                sys.exit(f"\n!! asked for seed {s}, env reset to {got[j]}.\n")
            batch_rows.append(dict(
                run_id=f"align_{tag}_{arm}", mode=tag, arm=arm,
                policy_seed=policy_seed, seed=s,
                cubeA_x=a0[j][0], cubeA_y=a0[j][1],
                cubeB_x=b0[j][0], cubeB_y=b0[j][1]))

        ever = {k: np.zeros(num_envs, bool) for k in STEP_FLAGS}
        steps, fin, done = 0, [None] * num_envs, False
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
            # GENERATED reward - the quantity this whole stage is about.
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
            # Yaw from the pose the ENVIRONMENT produced, never copied from the
            # T-II index row: a column copied from that record would match it
            # trivially, proving nothing about whether StackCube-T3-v1
            # reproduces the initial-state distribution.
            row["cubeA_theta"] = yaw(a0[j][3:7])
            row["cubeB_theta"] = yaw(b0[j][3:7])
            g = geom_features(row["cubeA_x"], row["cubeA_y"], row["cubeA_theta"],
                              row["cubeB_x"], row["cubeB_y"], row["cubeB_theta"])
            row.update({k: g[k] for k in
                        ("face_gap", "dist_A", "dist_B", "dist_max", "dist_min")})

        blank = [r["seed"] for r in batch_rows
                 if r["success_once"] == "" or r["ep_return"] == ""]
        if blank:
            sys.exit(
                f"\n!! {len(blank)} episode(s) came back with no metrics "
                f"(seeds {blank[:5]}...).\n"
                f"   CPUGymWrapper(record_metrics=True) nests 'return' and "
                f"'success_once'\n   under info['episode'], reached through "
                f"info['final_info']. Their absence\n   means the wrapper stack "
                f"is not what this expects - NOT that the reward\n   is zero. "
                f"Nothing written.\n")
        rows += batch_rows

        el = time.time() - t0
        k = sum(1 for r in rows if r["success_once"] == 1)
        print(f"    {arm:<7} {len(rows):>4}/{len(seeds)}  "
              f"{len(rows) / el * 3600:6.0f} ep/h   success {k / len(rows):.3f}   "
              f"mean return {mean(r['ep_return'] for r in rows):9.2f}", flush=True)
    return rows


def stage_align(a):
    # If a specific reward file was named, install it by pointing T3_RUN at a
    # directory holding it. env_t3 loads through the same path either way, so
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

    # Not negotiable, and recorded in the manifest so it can be checked rather
    # than taken on trust.
    os.environ["T3_SAMPLER"] = "0"
    os.environ["ENV_ID"] = "StackCube-T3-v1"

    label = a.label or a.mode
    os.makedirs(a.out, exist_ok=True)
    out_csv = os.path.join(a.out, f"align_{label}.csv")
    if os.path.exists(out_csv) and os.environ.get("FORCE", "0") != "1":
        with open(out_csv) as f:
            have = len(list(csv.DictReader(f)))
        want = a.episodes + a.control_episodes * (len(a.arms) - 1)
        if have >= want:
            print(f"  {out_csv} already has {have} rows, skipping (FORCE=1 to redo)")
            return
        sys.exit(f"\n!! {out_csv} has {have} rows, expected {want}. That is a "
                 f"crashed run,\n   not a finished one. Delete it or set "
                 f"FORCE=1; do not append to it.\n")

    seeds = load_seeds(a.mode, a.episodes)
    print(f"  {env_t3.describe()}")
    print(f"  seeds       {len(seeds)} from mode_{a.mode}_seed1.csv "
          f"[{min(seeds)}..{max(seeds)}]")

    agent, envs, args, device = build_agent(
        a.ckpt, a.state, a.num_envs, max_episode_steps=MAX_EP_STEPS,
        reward_mode="dense")

    rows, t0 = [], time.time()
    try:
        for arm in a.arms:
            n = a.episodes if arm == "policy" else a.control_episodes
            torch.manual_seed(a.policy_seed)
            np.random.seed(a.policy_seed)
            rows += run_arm(arm, agent, envs, device, a.mode, seeds[:n],
                            a.num_envs, a.policy_seed)
    finally:
        envs.close()

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ALIGN_COLUMNS)
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in ALIGN_COLUMNS} for r in rows])

    ns_path = a.reward or os.path.join(a.run, REWARD_FILE)
    reward_max = float(load_file(ns_path, "reward")[REWARD_MAX_NAME])
    with open(os.path.join(a.out, f"align_{label}_manifest.json"), "w") as f:
        json.dump(manifest(ckpt=a.ckpt, mode=a.mode, label=label,
                           reward=os.path.relpath(ns_path),
                           reward_max=reward_max,
                           t3_sampler=int(env_t3.sampler_enabled()),
                           arms=a.arms, episodes=a.episodes,
                           policy_seed=a.policy_seed,
                           seed_min=min(seeds), seed_max=max(seeds),
                           wall_seconds=round(time.time() - t0, 1)), f, indent=2)

    print(f"\n  wrote       {out_csv} ({len(rows)} episodes) and its manifest")


# ===========================================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sampler", "reward", "align"])
    ap.add_argument("--run")
    ap.add_argument("--mode", default="gap")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt")
    ap.add_argument("--reward", help="a reward file to use instead of the "
                                     "generation's - the stock calibration arm")
    ap.add_argument("--label", help="output label (default: mode)")
    ap.add_argument("--draws", type=int, default=4096)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--control-episodes", type=int, default=20)
    # policy only by default. jitter and zero are reachable but cost rollout
    # time to check a proposition the AUC already covers.
    ap.add_argument("--arms", nargs="+", default=["policy"])
    ap.add_argument("--num-envs", type=int, default=10)
    ap.add_argument("--state", action="store_true")
    ap.add_argument("--policy-seed", type=int, default=1)
    a = ap.parse_args()

    if a.stage == "align" and not a.ckpt:
        sys.exit("!! align needs --ckpt")
    if a.stage in ("sampler",) and not a.run:
        sys.exit("!! sampler needs --run")

    {"sampler": stage_sampler, "reward": stage_reward, "align": stage_align}[
        a.stage](a)
    print(f"\n  A measurement, not a verdict. `python3 t3/summary.py {a.out} "
          f"--mode {a.mode}` reads it.")


if __name__ == "__main__":
    main()
