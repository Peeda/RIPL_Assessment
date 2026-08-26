#!/usr/bin/env python
"""Layer E: measure the generated sampler. Needs torch; ManiSkill only for E2.

    python3 t3/sampler_check.py --run t3/artifacts/gap --mode gap \
                                --out $T3_OUT [--draws 4096] [--e1-only]

Deliverable (b) is "an episode-configuration sampler that biases initial
conditions toward the failure regime". This measures whether it does, and
whether the states it produces are ones the environment could actually have
produced itself.

E1 CALLS THE SAMPLER DIRECTLY, WITH NO SIMULATOR
    The contract makes sample_cube_poses(b, device) a pure function of a batch
    size and a device - it never touches `env`. That is not tidiness: it means
    4,096 draws can be checked in about a second, on a laptop, with torch and
    t2/geometry.py and nothing else. A sampler that had to be exercised through
    a running environment would cost a pod session per iteration, and iterating
    on the sampler is most of what tuning the prompt consists of.

E2 GOES THROUGH THE REAL ENVIRONMENT
    One property E1 cannot see: that reset(seed=s) is still reproducible once
    the sampler is installed. The environment seeds torch's generator and
    nothing else, so a sampler drawing from `random` or numpy would pass every
    E1 check and silently destroy the seed addressing that every evaluation in
    this project depends on. E2 resets twice at the same seed and compares.

    Honest scope: sapien_env.py seeds from _episode_seed[0] only, so at batch
    widths above one, row j is not a width-1 reset of seed j. That is already
    true of stock StackCube and is not new breakage - but it bounds the claim to
    "at a fixed batch width, reset(seed=s) is reproducible", which is the
    property T-II's physx_cpu (one env per subprocess) actually relies on.

THIS SCRIPT MEASURES; t3/verify.py DECIDES. Nothing here compares against a
threshold - it writes sampler_<tag>.csv (the raw draws, so report.py can plot
the biased distribution against the nominal one) and sampler_<tag>.json (the
statistics). The gate reads those. That split is what lets t3/test_verify.py
fabricate a collapsed sampler and check the gate catches it, with no torch and
no simulator anywhere.
"""
import argparse
import csv
import json
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(os.path.dirname(HERE), "t2"))
from geometry import MODES, geom_features, yaw  # noqa: E402
from loader import load_file  # noqa: E402
from spec import (CUBE_Z, MIN_SEPARATION, REACH_MAX, SAMPLER_COLUMNS,  # noqa: E402
                  SAMPLER_COVER_TOL, SAMPLER_YAW_BINS, SUPPORT_X, SUPPORT_Y,
                  SAMPLER_FILE, mean, sd)

T2_RESULTS = os.environ.get("T2_RESULTS",
                            os.path.join(os.path.dirname(HERE), "t2", "results"))


def base_rate(tag):
    """The nominal probability of the region, from the committed seed index.

    Computed at run time rather than hardcoded. The index is 8,000 policy-free
    resets of the stock environment, so this is the environment's own rate for
    the region - the number the sampler's hit rate has to be an enrichment over.
    """
    path = os.path.join(T2_RESULTS, "seeds.csv")
    if not os.path.exists(path):
        return None, 0
    from geometry import geom_from_row
    n = hit = 0
    with open(path) as f:
        for r in csv.DictReader(f):
            n += 1
            if MODES[tag](geom_from_row(r)):
                hit += 1
    return (hit / n if n else None), n


def eval_targets(tag):
    """The (face_gap, dist_A, dist_B) of the fixed T-II evaluation episodes.

    T-IV is scored on those exact seeds. A sampler can satisfy the region
    predicate while concentrating in one corner of it, which produces a residual
    that fits the corner and does not transfer - so coverage of the actual
    scoring set is measured, not assumed.
    """
    import glob
    from geometry import geom_from_row
    out = []
    for path in sorted(glob.glob(os.path.join(T2_RESULTS, f"mode_{tag}_seed*.csv"))):
        with open(path) as f:
            for r in csv.DictReader(f):
                g = geom_from_row(r)
                out.append((g["face_gap"], g["dist_A"], g["dist_B"]))
    return out


def draw(run, n, device, seed):
    torch.manual_seed(seed)
    ns = load_file(os.path.join(run, SAMPLER_FILE), "sampler")
    with torch.device(device):
        return ns["sample_cube_poses"](n, device)


def rows_from(out):
    """The sampler's tensors -> a list of plain dicts, via t2/geometry.py.

    Converted to floats immediately: everything downstream is geometry.py, which
    imports nothing but math, and the gate reads a CSV.
    """
    a_xyz = out["cubeA_xyz"].detach().cpu().tolist()
    b_xyz = out["cubeB_xyz"].detach().cpu().tolist()
    a_q = out["cubeA_quat"].detach().cpu().tolist()
    b_q = out["cubeB_quat"].detach().cpu().tolist()
    rows = []
    for i in range(len(a_xyz)):
        ta, tb = yaw(a_q[i]), yaw(b_q[i])
        g = geom_features(a_xyz[i][0], a_xyz[i][1], ta,
                          b_xyz[i][0], b_xyz[i][1], tb)
        rows.append(dict(
            i=i, cubeA_x=a_xyz[i][0], cubeA_y=a_xyz[i][1], cubeA_theta=ta,
            cubeB_x=b_xyz[i][0], cubeB_y=b_xyz[i][1], cubeB_theta=tb,
            separation=math.dist((a_xyz[i][0], a_xyz[i][1]),
                                 (b_xyz[i][0], b_xyz[i][1])),
            _az=a_xyz[i][2], _bz=b_xyz[i][2], _aq=a_q[i], _bq=b_q[i], **g))
    return rows


def measure(rows, tag, out_raw):
    """Every layer-E statistic, as raw numbers. No thresholds live here."""
    n = len(rows)
    quat_bad = sum(1 for r in rows
                   for q in (r["_aq"], r["_bq"])
                   if abs(math.sqrt(sum(v * v for v in q)) - 1) > 1e-5
                   or abs(q[1]) > 1e-6 or abs(q[2]) > 1e-6)
    z_bad = sum(1 for r in rows
                for z in (r["_az"], r["_bz"]) if abs(z - CUBE_Z) > 1e-6)
    sep_bad = sum(1 for r in rows if r["separation"] < MIN_SEPARATION - 1e-9)
    support_bad = sum(
        1 for r in rows
        if not (SUPPORT_X[0] - 1e-6 <= r["cubeA_x"] <= SUPPORT_X[1] + 1e-6
                and SUPPORT_X[0] - 1e-6 <= r["cubeB_x"] <= SUPPORT_X[1] + 1e-6
                and SUPPORT_Y[0] - 1e-6 <= r["cubeA_y"] <= SUPPORT_Y[1] + 1e-6
                and SUPPORT_Y[0] - 1e-6 <= r["cubeB_y"] <= SUPPORT_Y[1] + 1e-6))
    reach_bad = sum(1 for r in rows
                    if r["dist_A"] > REACH_MAX or r["dist_B"] > REACH_MAX)

    hits = sum(1 for r in rows if MODES[tag](r))
    br, br_n = base_rate(tag)

    # Non-degeneracy. A sampler collapsed onto one configuration hits the region
    # perfectly and teaches a residual a single initial state.
    bins_a = {int((r["cubeA_theta"] + math.pi) / (2 * math.pi) * SAMPLER_YAW_BINS)
              % SAMPLER_YAW_BINS for r in rows}
    bins_b = {int((r["cubeB_theta"] + math.pi) / (2 * math.pi) * SAMPLER_YAW_BINS)
              % SAMPLER_YAW_BINS for r in rows}
    distinct = len({(round(r["cubeA_x"], 3), round(r["cubeA_y"], 3),
                     round(r["cubeB_x"], 3), round(r["cubeB_y"], 3)) for r in rows})

    targets = eval_targets(tag)
    covered = None
    if targets:
        pts = [(r["face_gap"], r["dist_A"], r["dist_B"]) for r in rows]
        c = 0
        for t in targets:
            if any(abs(p[0] - t[0]) <= SAMPLER_COVER_TOL
                   and abs(p[1] - t[1]) <= SAMPLER_COVER_TOL
                   and abs(p[2] - t[2]) <= SAMPLER_COVER_TOL for p in pts):
                c += 1
        covered = c / len(targets)

    return dict(
        mode=tag, draws=n,
        quat_bad=quat_bad, z_bad=z_bad, separation_bad=sep_bad,
        support_bad=support_bad, reach_bad=reach_bad,
        hit_rate=hits / n, hits=hits,
        base_rate=br, base_rate_n=br_n,
        enrichment=(hits / n / br) if br else None,
        sd_cubeA_x=sd(r["cubeA_x"] for r in rows),
        sd_cubeA_y=sd(r["cubeA_y"] for r in rows),
        sd_cubeB_x=sd(r["cubeB_x"] for r in rows),
        sd_cubeB_y=sd(r["cubeB_y"] for r in rows),
        sd_face_gap=sd(r["face_gap"] for r in rows),
        yaw_bins_A=len(bins_a), yaw_bins_B=len(bins_b),
        distinct_frac=distinct / n,
        mean_face_gap=mean(r["face_gap"] for r in rows),
        mean_dist_A=mean(r["dist_A"] for r in rows),
        mean_dist_B=mean(r["dist_B"] for r in rows),
        eval_seed_coverage=covered, eval_seed_n=len(targets),
        deterministic=out_raw,
    )


def determinism(run, device, n=256):
    """Same torch seed, twice. -> True/False.

    The one property that catches a sampler drawing from `random` or numpy,
    which every other E1 check would pass.
    """
    a = draw(run, n, device, 12345)
    b = draw(run, n, device, 12345)
    return all(torch.equal(a[k], b[k]) for k in a)


def e2_env_determinism(seeds=(10001, 10002, 10003, 10004)):
    """reset(seed=s) twice through the real environment, at width 1."""
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    import env_t3  # noqa: F401  - registers StackCube-T3-v1

    env = gym.make("StackCube-T3-v1", num_envs=1, obs_mode="state",
                   sim_backend="physx_cpu", reconfiguration_freq=0,
                   control_mode=os.environ.get("CTRL", "pd_ee_delta_pos"))
    ok = True
    try:
        for s in seeds:
            env.reset(seed=s)
            a1 = env.unwrapped.cubeA.pose.raw_pose.clone()
            b1 = env.unwrapped.cubeB.pose.raw_pose.clone()
            env.reset(seed=s)
            if not (torch.equal(a1, env.unwrapped.cubeA.pose.raw_pose)
                    and torch.equal(b1, env.unwrapped.cubeB.pose.raw_pose)):
                ok = False
    finally:
        env.close()
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=4096)
    ap.add_argument("--e1-only", action="store_true")
    a = ap.parse_args()

    device = "cpu"
    os.makedirs(a.out, exist_ok=True)
    print(f"  sampler     {a.run}/{SAMPLER_FILE}, mode '{a.mode}', "
          f"{a.draws} draws")

    out = draw(a.run, a.draws, device, 0)
    rows = rows_from(out)
    det = determinism(a.run, device)
    m = measure(rows, a.mode, det)

    if not a.e1_only:
        # Needs a ManiSkill install and the biased sampler switched on, so it is
        # skipped on the laptop rather than failing the whole stage.
        prev = os.environ.get("T3_SAMPLER")
        os.environ["T3_SAMPLER"] = "1"
        try:
            m["env_determinism"] = e2_env_determinism()
        except Exception as e:                                   # noqa: BLE001
            m["env_determinism"] = None
            m["env_determinism_error"] = f"{type(e).__name__}: {e}"
            print(f"  E2 skipped  {type(e).__name__}: {e}")
        finally:
            if prev is None:
                os.environ.pop("T3_SAMPLER", None)
            else:
                os.environ["T3_SAMPLER"] = prev

    csv_path = os.path.join(a.out, f"sampler_{a.mode}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SAMPLER_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (int(MODES[a.mode](r)) if k == "in_region" else r[k])
                        for k in SAMPLER_COLUMNS})
    with open(os.path.join(a.out, f"sampler_{a.mode}.json"), "w") as f:
        json.dump(m, f, indent=2)

    br = f"{m['base_rate']:.4f}" if m["base_rate"] else "?"
    enr = f"{m['enrichment']:.1f}x" if m["enrichment"] else "?"
    print(f"  hit rate    {m['hit_rate']:.3f}  (nominal {br} over "
          f"{m['base_rate_n']} indexed seeds -> {enr})")
    print(f"  invalid     quat {m['quat_bad']}  z {m['z_bad']}  "
          f"separation {m['separation_bad']}  support {m['support_bad']}  "
          f"reach {m['reach_bad']}")
    print(f"  spread      sd(A) {m['sd_cubeA_x']:.3f}/{m['sd_cubeA_y']:.3f}  "
          f"sd(gap) {m['sd_face_gap']:.4f}  yaw bins {m['yaw_bins_A']}/"
          f"{m['yaw_bins_B']} of {SAMPLER_YAW_BINS}  "
          f"distinct {m['distinct_frac']:.2f}")
    if m["eval_seed_coverage"] is not None:
        print(f"  coverage    {m['eval_seed_coverage']:.2f} of "
              f"{m['eval_seed_n']} T-II evaluation episodes within "
              f"{SAMPLER_COVER_TOL * 1000:.0f} mm")
    print(f"  wrote       {csv_path} and sampler_{a.mode}.json")
    print(f"\n  These are measurements, not a verdict. t3/verify.py applies the "
          f"thresholds.")


if __name__ == "__main__":
    main()
