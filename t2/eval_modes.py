#!/usr/bin/env python
"""Per-failure-mode success rate: 100 rollouts x 3 seeds, per mode.

    source /workspace/ripl/env.sh
    python t2/eval_modes.py CKPT [--state] [--modes nominal gap farb]
                            [--index seeds.csv] [--out DIR]
                            [--episodes 100] [--blocks 3] [--num-envs 10]

This is the T-II deliverable. The assignment asks for "per-failure-mode success
rates of the base policy over 100 rollouts and 3 seeds by running evaluations in
the identified failure episode configurations", and this produces exactly that,
for both modes plus a nominal reference arm in the identical shape.

WHAT "100 ROLLOUTS x 3 SEEDS" MEANS HERE
    Three DISJOINT blocks of 100 region seeds, block b run under policy seed b.
    The three rates are independent estimates and their spread is a real error
    bar. Reusing one block of 100 and running it three times under three policy
    seeds would hold the initial states fixed and measure only DDPM sampling
    noise, which is a much smaller quantity - and not what "3 seeds" means.

WHY THE SEEDS ARE THE WHOLE TRICK
    reset(seed=s) is deterministic, so a seed is a lossless 8-byte encoding of
    the entire initial state. seed_index.py tabulates seed -> cube poses with
    resets alone, and "resample fresh episodes from the failure region" becomes
    rejection sampling over integers. The episodes that come back are drawn from
    exactly the environment's own conditional distribution given the region: no
    state injection, no distribution shift, every episode reproducible from one
    integer. See t2/README.md.

WRITTEN TO A FILE AND RUN, never piped in as a heredoc: forkserver re-imports
__main__ in every child, and under `python - <<PY` it is not importable. The
child dies during handshake and the parent reports ConnectionResetError, which
reads like a pod fault and is not one. Hence the __main__ guard and
module-level imports. See CLAUDE.md's traps.
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry import (COLUMNS, EVAL_BASE, MODES, cube_features,  # noqa: E402
                      geom_from_row, reserved_hit, wilson)
from harness import (attach_residual, build_agent, flag,  # noqa: E402
                     manifest, poses_from_info, residual_path, sha16,
                     to_device)

MAX_EP_STEPS = int(os.environ.get("MAX_EP_STEPS", 200))

# The four flags StackCubeEnv.evaluate() returns. `success` is the conjunction;
# the other three are the stage decomposition, and logging them is what turns a
# binary failure into a mechanism - "never grasped" and "grasped but never
# placed" and "placed but did not hold" are genuinely different failures, and
# separating them is most of what T-II is graded on.
STEP_FLAGS = ("success", "is_cubeA_grasped", "is_cubeA_on_cubeB", "is_cubeA_static")

# ---------------------------------------------------------------------------
# seed selection
# ---------------------------------------------------------------------------


def select_seeds(index_path, modes, n_per_mode):
    """{mode: [seed, ...]} - disjoint by construction, and provably so.

    Allocated from ONE pool above EVAL_BASE in a fixed mode order, each chosen
    seed removed as it goes. Two passes cannot share an episode, and nothing can
    land in a reserved block, because the pool never contained one.

    The old harness selected per-region from `seed >= 2200` and needed ~5,000
    eligible seeds to fill a 300-seed region - which reached straight through
    T-I's [6000, 6300) block and silently pulled in 20 of its seeds. A shared,
    shrinking pool above everything already measured removes the failure mode
    rather than checking for it afterwards.
    """
    rows = list(csv.DictReader(open(index_path)))
    if not rows:
        sys.exit(f"!! {index_path} is empty - run 'bash t2/run.sh index' first")

    pool = []
    for r in rows:
        s = int(r["seed"])
        if s < EVAL_BASE or reserved_hit(s):
            continue
        pool.append((s, geom_from_row(r)))
    print(f"  index      {len(rows)} seeds, {len(pool)} eligible "
          f"(>= {EVAL_BASE}, outside every reserved block)")

    taken, out = set(), {}
    for tag in modes:
        pred = MODES[tag]
        hits = [s for s, g in pool if s not in taken and pred(g)]
        rate = len(hits) / max(len(pool), 1)
        print(f"  {tag:10s} {len(hits):5d} available  ({rate:6.2%} of the pool)")
        if len(hits) < n_per_mode:
            sys.exit(
                f"\n!! mode '{tag}' has only {len(hits)} eligible seeds, needs "
                f"{n_per_mode}.\n"
                f"   At {rate:.2%} this needs about "
                f"{EVAL_BASE + int(n_per_mode / max(rate, 1e-9)):,} indexed seeds.\n"
                f"   Extend the index - indexing is policy-free and GPU-free, and\n"
                f"   is much cheaper than relaxing a pre-registered threshold to\n"
                f"   fit the index you happen to have.\n")
        out[tag] = hits[:n_per_mode]
        taken |= set(out[tag])
    return out


# ---------------------------------------------------------------------------
# the rollout
# ---------------------------------------------------------------------------


def check_episode(tag, seed, a_pose, b_pose, got_seed, index_row, used):
    """The four assertions, run at reset before a single step is taken.

    This is the "verifiably correct" requirement, and it lives in the loop
    rather than in a post-hoc script for one reason: an episode that fails any
    of these must never be LOGGED, because a CSV row is what a report quotes.
    verify.py re-runs checks 1-3 offline from the committed CSVs; this is what
    stops a bad row existing in the first place.

    Returns the feature dict for the episode. Exits naming the seed on failure.
    """
    def die(msg):
        sys.exit(f"\n!! mode '{tag}', seed {seed}: {msg}\n"
                 f"   Refusing to log an episode that is not provably what it "
                 f"claims to be.\n")

    # 1. The env reset to the seed we asked for. If a seed silently fails to
    #    take, every episode downstream describes a different initial state
    #    than the one the region was selected on.
    if got_seed != seed:
        die(f"env reset to seed {got_seed}, not {seed}")

    # 2. The seed is outside every reserved block and unused elsewhere in this
    #    run. select_seeds guarantees both; asserting it here means a future
    #    change to selection cannot quietly break disjointness.
    hit = reserved_hit(seed)
    if hit:
        die(f"seed is inside the reserved '{hit}' block")
    if seed in used:
        die("seed already used by another block in this run")

    feats = cube_features(a_pose, b_pose)

    # 3. The poses read out of THIS env match the independently-built index.
    #    seed_index.py and this script touch the simulator by different paths;
    #    agreement means the seed -> state map is genuine and deterministic.
    if index_row is not None:
        for k in ("cubeA_x", "cubeA_y", "cubeA_theta",
                  "cubeB_x", "cubeB_y", "cubeB_theta"):
            d = abs(feats[k] - float(index_row[k]))
            if d > 1e-5:
                die(f"{k} is {feats[k]:.6f} here but {float(index_row[k]):.6f} "
                    f"in the seed index (diff {d:.2e})")

    # 4. The episode is IN THE REGION - recomputed from what this env actually
    #    produced, not from the index row that selected it. Without this, a
    #    selection bug would file episodes under a mode they do not belong to
    #    and the per-mode rate would be a rate over the wrong population.
    if not MODES[tag](feats):
        die(f"initial state is outside mode '{tag}'  "
            f"(face_gap {feats['face_gap'] * 1000:.1f} mm, "
            f"dist_A {feats['dist_A'] * 1000:.0f} mm, "
            f"dist_B {feats['dist_B'] * 1000:.0f} mm)")

    return feats


def run_block(agent, envs, device, tag, block, seeds, index, used, num_envs):
    """One block: `seeds` episodes under policy seed `block`. -> list of rows."""
    rows = []
    t0 = time.time()
    for start in range(0, len(seeds), num_envs):
        batch = seeds[start:start + num_envs]
        # The last batch can be short. Pad by repeating the final seed rather
        # than rebuilding the vector env at a smaller width, and drop the
        # padding rows afterwards.
        bseeds = batch + [batch[-1]] * (num_envs - len(batch))

        # Drop any unconsumed base chunk. Only reachable when T-IV runs a
        # residual with res_horizon < act_horizon AND the previous episode ended
        # mid-chunk; carrying that plan into a fresh initial state would apply
        # the last episode's actions to this one's cubes.
        agent.reset_chunk()

        obs, info = envs.reset(seed=bseeds)
        n = len(batch)
        a0 = poses_from_info(info, "cubeA_pose", n)
        b0 = poses_from_info(info, "cubeB_pose", n)
        got = [int(np.asarray(v).reshape(-1)[0])
               for v in np.atleast_1d(info["episode_seed"])[:n]]

        batch_rows = []
        for j, s in enumerate(batch):
            feats = check_episode(tag, s, a0[j], b0[j], got[j],
                                  index.get(s), used)
            used.add(s)
            batch_rows.append(dict(
                run_id=f"mode_{tag}_seed{block}", mode=tag, block=block,
                policy_seed=block, seed=s,
                **{k: feats[k] for k in COLUMNS if k in feats}))

        # per-episode accumulators - this is what evaluate() throws away
        ever = {k: np.zeros(num_envs, bool) for k in STEP_FLAGS}
        steps, fin = 0, [None] * num_envs
        done = False
        while not done and steps < MAX_EP_STEPS:
            with torch.no_grad():
                chunk = agent.get_action(to_device(obs, device)).cpu().numpy()
            for i in range(chunk.shape[1]):
                obs, rew, term, trunc, info = envs.step(chunk[:, i])
                steps += 1
                for k in STEP_FLAGS:
                    if k in info:
                        v = np.asarray([bool(np.asarray(x).reshape(-1)[0])
                                        for x in np.atleast_1d(info[k])], bool)
                        ever[k] |= v
                if trunc.any() or term.any():
                    done = True
                    # CPUGymWrapper(ignore_terminations=True, record_metrics=True)
                    # runs every episode to the full horizon and nests the real
                    # metrics under info['episode']. The top-level 'success' is
                    # the final-step value - success_at_end - NOT the
                    # success_once CLAUDE.md fixes as the reported number.
                    fin = list(np.atleast_1d(info.get("final_info", info)))
                    break

        for j, row in enumerate(batch_rows):
            fi = fin[j] if isinstance(fin[j], dict) else {}
            epi = fi.get("episode", {}) if isinstance(fi, dict) else {}
            # Terminal poses come from final_info, NOT from the top-level info.
            # ignore_terminations=True runs every episode to the horizon, where
            # AsyncVectorEnv autoresets - so at the step that ends the episode
            # the top-level poses are already the NEXT episode's initial state.
            # Reading them there would silently make cubeB_displacement the
            # distance between two unrelated episodes' cubes.
            af = [float(v) for v in np.asarray(
                fi.get("cubeA_pose", [np.nan] * 7), float).reshape(-1)]
            bf = [float(v) for v in np.asarray(
                fi.get("cubeB_pose", [np.nan] * 7), float).reshape(-1)]
            row["success_once"] = flag(epi, "success_once")
            row["success_at_end"] = flag(epi, "success_at_end")
            row["ep_len"] = int(np.asarray(epi.get("episode_len", steps)).reshape(-1)[0]) \
                if epi else steps
            row["ever_grasped"] = int(ever["is_cubeA_grasped"][j])
            row["ever_placed"] = int(ever["is_cubeA_on_cubeB"][j])
            row["ever_static"] = int(ever["is_cubeA_static"][j])
            row["final_cubeA_x"], row["final_cubeA_y"], row["final_cubeA_z"] = af[:3]
            # How far cubeB was shoved. Mode A's mechanism is that the descent
            # onto cubeA fouls cubeB; this measures it directly, turning a
            # correlation into a mechanism for the price of one column.
            row["cubeB_displacement"] = float(
                np.hypot(bf[0] - b0[j][0], bf[1] - b0[j][1]))
        # A missing metric is silently a FAILURE downstream: flag() returns ''
        # when the key is absent, '' != '1', and the episode would be counted as
        # unsuccessful. That turns a broken info path - the wrong wrapper order,
        # a gymnasium version that moved final_info - into a plausible-looking
        # low success rate rather than an error. Refuse to log it.
        blank = [r["seed"] for r in batch_rows if r["success_once"] == ""]
        if blank:
            sys.exit(
                f"\n!! {len(blank)} episode(s) came back with no success_once "
                f"(seeds {blank[:5]}...).\n"
                f"   CPUGymWrapper(record_metrics=True) nests it under "
                f"info['episode'], reached\n"
                f"   through info['final_info'] at the truncating step. Its "
                f"absence means the\n"
                f"   wrapper stack or the gymnasium version is not what this "
                f"harness expects -\n"
                f"   NOT that the policy failed. Nothing has been written.\n")
        rows += batch_rows

        el = time.time() - t0
        k = sum(1 for r in rows if r["success_once"] == 1)
        print(f"    {len(rows):>4}/{len(seeds)}  {len(rows) / el * 3600:6.0f} ep/h"
              f"   success_once so far {k / len(rows):.3f}", flush=True)
    return rows


# ---------------------------------------------------------------------------


def summarise(by_mode):
    """The deliverable table. Also printed by verify.py from the CSVs alone."""
    print(f"\n{'mode':>9} {'per-block rates':>22} {'mean':>7} {'SD':>7} "
          f"{'pooled 95% CI':>16} {'grasp':>7} {'place':>7} {'hold|pl':>8}")
    for tag, blocks in by_mode.items():
        rates = [sum(1 for r in b if r["success_once"] == 1) / len(b)
                 for b in blocks]
        rows = [r for b in blocks for r in b]
        k = sum(1 for r in rows if r["success_once"] == 1)
        n = len(rows)
        m = sum(rates) / len(rates)
        sd = (sum((x - m) ** 2 for x in rates) / (len(rates) - 1)) ** 0.5 \
            if len(rates) > 1 else float("nan")
        lo, hi = wilson(k, n)
        g = sum(r["ever_grasped"] for r in rows) / n
        p = sum(r["ever_placed"] for r in rows) / n
        held = k / max(sum(r["ever_placed"] for r in rows), 1)
        print(f"{tag:>9} {' '.join(f'{x:.3f}' for x in rates):>22} "
              f"{m:7.3f} {sd:7.3f} [{lo:.3f}, {hi:.3f}] "
              f"{g:7.3f} {p:7.3f} {held:8.3f}")
    print("\n  success_once over n=100 x 3 blocks. SD is over the three blocks,"
          "\n  each a disjoint set of 100 region seeds under its own policy seed."
          "\n  grasp/place are ever_* rates; hold|pl is success given placement -"
          "\n  the column that separates the two modes' mechanisms.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--state", action="store_true")
    p.add_argument("--modes", nargs="+", default=["nominal", "gap", "farb"],
                   choices=sorted(MODES))
    p.add_argument("--index", default=None, help="seeds.csv from seed_index.py")
    p.add_argument("--out", default=".")
    p.add_argument("--episodes", type=int, default=100, help="per block")
    p.add_argument("--blocks", type=int, default=3, help="= policy seeds")
    p.add_argument("--num-envs", type=int, default=10)
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)
    index_path = a.index or os.path.join(a.out, "seeds.csv")
    n_per_mode = a.episodes * a.blocks

    print(f"per-mode evaluation: {a.episodes} rollouts x {a.blocks} seeds, "
          f"modes {', '.join(a.modes)}")
    seeds = select_seeds(index_path, a.modes, n_per_mode)
    index = {int(r["seed"]): r for r in csv.DictReader(open(index_path))}

    agent, envs, args, device = build_agent(a.ckpt, a.state, a.num_envs,
                                            max_episode_steps=MAX_EP_STEPS)
    meta = manifest(ckpt=os.path.abspath(a.ckpt),
                    obs_mode="state" if a.state else "rgb",
                    episodes_per_block=a.episodes, blocks=a.blocks,
                    num_envs=a.num_envs, max_episode_steps=MAX_EP_STEPS)
    for k, v in meta.items():
        print(f"  {k:16s} {v}")

    # Shared across every mode and block, so disjointness is enforced globally
    # rather than per-pass.
    used = set()
    # Residual provenance for the block currently loaded. Empty for the base
    # policy, so every pre-T-IV manifest keeps exactly the keys it had.
    block_meta = {}
    by_mode, t0 = {}, time.time()
    for tag in a.modes:
        blocks = []
        for b in range(1, a.blocks + 1):
            lo = (b - 1) * a.episodes
            block_seeds = seeds[tag][lo:lo + a.episodes]
            prefix = os.path.join(a.out, f"mode_{tag}_seed{b}")
            print(f"\n=== mode '{tag}' block {b}/{a.blocks} - {len(block_seeds)} "
                  f"episodes, policy seed {b}  [{time.time() - t0:.0f}s]")

            # Never overwrite a finished block. Rollouts are stochastic, so
            # anything clobbered here is gone for good.
            if os.path.exists(prefix + ".csv") and os.environ.get("FORCE") != "1":
                rows = list(csv.DictReader(open(prefix + ".csv")))
                # A SHORT block is a crashed block, not a finished one, and
                # resuming past it would silently report a rate over the wrong
                # sample size. The likeliest way to get one is a smoke run or an
                # interrupted job writing into this directory - both plausible,
                # neither something to paper over.
                if len(rows) != a.episodes:
                    sys.exit(
                        f"\n!! {prefix}.csv has {len(rows)} episodes, expected "
                        f"{a.episodes}.\n"
                        f"   That is an interrupted or differently-sized run, not "
                        f"a finished block.\n"
                        f"   Delete it and re-run, or set FORCE=1 to overwrite it "
                        f"(the existing\n"
                        f"   rollouts are stochastic and will not come back).\n")
                for r in rows:
                    for k in ("success_once", "ever_grasped", "ever_placed",
                              "ever_static"):
                        r[k] = int(r[k])
                    r["seed"] = int(r["seed"])
                used |= {r["seed"] for r in rows}
                blocks.append(rows)
                print(f"    skip - {prefix}.csv exists ({len(rows)} episodes). "
                      f"FORCE=1 to re-run.")
                continue

            # T-IV pairs residual seed b with block b, so the head changes here
            # while the base, the env and the seed selection do not. build_agent
            # has already attached a non-templated $RESIDUAL, and this is a
            # no-op when there is none.
            rp = residual_path(b)
            if rp:
                rm = attach_residual(agent, rp, device)
                block_meta = dict(
                    residual=rp, residual_sha256=sha16(rp),
                    **{f"residual_{k}": rm[k] for k in
                       ("mode", "seed", "alpha", "res_horizon") if k in rm})

            # Policy seed b. Set immediately before the block so each block's
            # DDPM sampling is reproducible from its own seed.
            torch.manual_seed(b)
            np.random.seed(b)
            rows = run_block(agent, envs, device, tag, b, block_seeds, index,
                             used, a.num_envs)

            with open(prefix + ".csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=COLUMNS)
                w.writeheader()
                w.writerows(rows)
            with open(prefix + "_manifest.json", "w") as f:
                json.dump(dict(meta, **block_meta, mode=tag, block=b, policy_seed=b,
                               episodes=len(rows),
                               seed_min=min(block_seeds), seed_max=max(block_seeds),
                               wall_seconds=round(time.time() - t0, 1)), f, indent=2)
            blocks.append(rows)
        by_mode[tag] = blocks

    envs.close()
    summarise(by_mode)
    print(f"\nwrote {a.out}/mode_<tag>_seed<b>.csv  "
          f"({sum(len(b) for bl in by_mode.values() for b in bl)} episodes in "
          f"{(time.time() - t0) / 60:.1f} min)")
    print("\n  Now run:  python t2/verify.py " + a.out)
    print("  The numbers are not reportable until it exits 0.")


if __name__ == "__main__":
    main()
