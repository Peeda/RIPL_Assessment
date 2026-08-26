#!/usr/bin/env python
"""Fabricate a valid T-III pass, break it seventeen ways, check the gate notices.

    python3 t3/test_verify.py

No deps, no simulator, no API key, ~2 s.

THE POINT. t3/verify.py is the only thing standing between a plausible-looking
LLM-generated reward and a GPU-day spent training on it. Every other test in
this directory checks that something is computed correctly; this one checks that
the gate REJECTS - which is the property that actually matters and the one that
is invisible until you deliberately break something.

It is only possible because the sim-touching scripts write numbers to files and
verify.py reads them. That split means a fake pass is a handful of CSVs, and a
corruption is one edited cell. Sixteen of the cases below correspond to a
specific way an LLM-written reward or sampler goes wrong; the seventeenth is a
truncated run, which is how a crashed pod job presents itself.

Structure copied from t2/test_verify.py, which does the same job for T-II.
"""
import csv
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from spec import ALIGN_COLUMNS, PROBE_COLUMNS, SWEEP_COLUMNS  # noqa: E402

MODE = "gap"
FAILS = []


def check(ok, label, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILS.append(label)


# ---------------------------------------------------------------------------
# a valid pass
# ---------------------------------------------------------------------------
#
# The stage counts are T-II's measured ones for mode `gap` at n = 100: 96
# grasped, 66 placed, 52 successful. Using the real rates matters - a fake pass
# built on 50/50 outcomes would give every threshold more headroom than it has
# in practice, and the test would stop reflecting the situation the gate meets.
STAGES = [("none", 4, 200.0), ("grasped", 30, 400.0),
          ("placed", 14, 700.0), ("success", 52, 1000.0)]

PROBE_R = {"P0_success": 8.0, "P7_held": 5.0, "P1_hover": 3.0,
           "P2_adjacent": 2.0, "P6_far": 1.2, "P3_knocked": 1.5,
           "P4_inverted": 1.0, "P5_offtable": 0.5, "P8_start": 0.8}


def _align_rows(rnd, seeds):
    rows, i = [], 0
    for name, n, base in STAGES:
        for _ in range(n):
            s = seeds[i]
            i += 1
            rows.append(dict(
                run_id=f"align_{MODE}_policy", mode=MODE, arm="policy",
                policy_seed=1, seed=s,
                cubeA_x=round(0.01 * (s % 17) - 0.08, 6),
                cubeA_y=round(0.01 * (s % 23) - 0.11, 6),
                cubeA_theta=round(0.001 * (s % 600) - 0.3, 6),
                cubeB_x=round(0.01 * (s % 19) - 0.09, 6),
                cubeB_y=round(0.01 * (s % 29) - 0.14, 6),
                cubeB_theta=round(0.001 * (s % 700) - 0.35, 6),
                face_gap=0.012, dist_A=0.62, dist_B=0.68,
                dist_max=0.68, dist_min=0.62,
                ep_return=round(base + rnd.uniform(-40, 40), 3),
                ep_reward_mean=round(base / 200, 4), ep_len=200,
                success_once=int(name == "success"),
                success_at_end=int(name == "success"),
                ever_grasped=int(name != "none"),
                ever_placed=int(name in ("placed", "success")),
                ever_static=int(name == "success")))
    for arm, base, n in (("jitter", 220.0, 20), ("zero", 120.0, 20)):
        for j in range(n):
            s = seeds[j]
            r = dict(rows[j])
            r.update(run_id=f"align_{MODE}_{arm}", arm=arm,
                     ep_return=round(base + rnd.uniform(-30, 30), 3),
                     ep_reward_mean=round(base / 200, 4),
                     success_once=0, success_at_end=0,
                     ever_grasped=0, ever_placed=0, ever_static=0)
            rows.append(r)
    return rows


def build(root):
    """A complete, internally consistent T-III validation output."""
    rnd = random.Random(7)
    out = os.path.join(root, "out")
    run = os.path.join(root, "run")
    t2 = os.path.join(root, "t2results")
    for d in (out, run, t2):
        os.makedirs(d, exist_ok=True)

    seeds = list(range(10000, 10100))
    rows = _align_rows(rnd, seeds)

    with open(os.path.join(out, f"align_{MODE}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ALIGN_COLUMNS)
        w.writeheader()
        w.writerows([{k: r[k] for k in ALIGN_COLUMNS} for r in rows])

    # T-II's record of the same seeds. The gate joins on all six pose columns,
    # so these have to agree exactly with what the align rows claim.
    pol = [r for r in rows if r["arm"] == "policy"]
    with open(os.path.join(t2, f"mode_{MODE}_seed1.csv"), "w", newline="") as f:
        cols = ["seed", "cubeA_x", "cubeA_y", "cubeA_theta",
                "cubeB_x", "cubeB_y", "cubeB_theta"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows([{k: r[k] for k in cols} for r in pol])

    with open(os.path.join(out, f"align_{MODE}_manifest.json"), "w") as f:
        json.dump(dict(ckpt_sha256="abc123", mode=MODE, label=MODE,
                       reward="run/reward.py", reward_max=8.0, t3_sampler=0,
                       arms=["policy", "jitter", "zero"], episodes=100), f)

    with open(os.path.join(out, f"probes_{MODE}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PROBE_COLUMNS)
        w.writeheader()
        for pid, val in PROBE_R.items():
            w.writerow(dict(
                probe=pid, reward=val, reward_norm=val / 8.0,
                is_cubeA_grasped=int(pid == "P7_held"),
                is_cubeA_on_cubeB=int(pid == "P0_success"),
                is_cubeA_static=int(pid == "P0_success"),
                success=int(pid == "P0_success"),
                cubeA_x=0.0, cubeA_y=0.0, cubeA_z=0.06,
                cubeB_x=0.0, cubeB_y=0.0, cubeB_z=0.02))

    with open(os.path.join(out, f"sweeps_{MODE}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SWEEP_COLUMNS)
        w.writeheader()
        for name in ("z", "r"):
            for i in range(24):
                w.writerow(dict(sweep=name, i=i, offset=0.01 * i,
                                reward=round(8.0 - 0.3 * i, 4)))

    with open(os.path.join(out, f"probes_{MODE}.json"), "w") as f:
        json.dump(dict(mode=MODE, label=MODE, reward="run/reward.py",
                       reward_max=8.0, notes={},
                       layer_b=dict(shape_ok=True, dtype_ok=True, finite_ok=True,
                                    device_ok=True, bounds_ok=True, pure_ok=True,
                                    mutation_ok=True, r_min=0.0, r_max=8.0,
                                    n_calls=320, reward_max=8.0,
                                    touched=["cubeA", "cubeB", "agent"],
                                    bad_attrs=[])), f)

    with open(os.path.join(out, f"sampler_{MODE}.json"), "w") as f:
        json.dump(dict(mode=MODE, draws=4096, quat_bad=0, z_bad=0,
                       separation_bad=0, support_bad=0, reach_bad=0,
                       hit_rate=0.95, hits=3891, base_rate=0.0464,
                       base_rate_n=8000, enrichment=20.5,
                       sd_cubeA_x=0.05, sd_cubeA_y=0.09, sd_cubeB_x=0.05,
                       sd_cubeB_y=0.09, sd_face_gap=0.008,
                       yaw_bins_A=12, yaw_bins_B=12, distinct_frac=1.0,
                       mean_face_gap=0.010, mean_dist_A=0.62, mean_dist_B=0.66,
                       eval_seed_coverage=0.92, eval_seed_n=300,
                       deterministic=True, env_determinism=True), f)

    shutil.copyfile(os.path.join(HERE, "fixtures", "good_reward.py"),
                    os.path.join(run, "reward.py"))
    shutil.copyfile(os.path.join(HERE, "fixtures", "good_sampler.py"),
                    os.path.join(run, "sampler.py"))
    return out, run, t2


def run_gate(out, run, t2):
    p = subprocess.run(
        [sys.executable, os.path.join(HERE, "verify.py"), out,
         "--mode", MODE, "--run", run, "--t2-results", t2],
        capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


# ---------------------------------------------------------------------------
# the corruptions
# ---------------------------------------------------------------------------


def _edit_csv(path, fn):
    rows = list(csv.DictReader(open(path)))
    cols = list(rows[0].keys())
    rows = fn(rows)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _edit_json(path, fn):
    j = json.load(open(path))
    json.dump(fn(j), open(path, "w"))


def _probe(out, pid, val):
    def f(rows):
        for r in rows:
            if r["probe"] == pid:
                r["reward"] = val
                r["reward_norm"] = float(val) / 8.0
        return rows
    _edit_csv(os.path.join(out, f"probes_{MODE}.csv"), f)


def _align(out, fn):
    _edit_csv(os.path.join(out, f"align_{MODE}.csv"), fn)


CORRUPTIONS = [
    ("a cube held above the target outscores a completed stack",
     lambda o, r, t: _probe(o, "P7_held", 9.0), "P0_success > P7_held"),

    ("the goal arguments are swapped - cubeB on cubeA scores highest",
     lambda o, r, t: _probe(o, "P4_inverted", 9.0), "P0_success > P4_inverted"),

    ("merely hovering over the target scores as well as stacking",
     lambda o, r, t: _probe(o, "P1_hover", 7.9), "P0_success > P1_hover"),

    ("the reference state is not actually a success",
     lambda o, r, t: _edit_csv(
         os.path.join(o, f"probes_{MODE}.csv"),
         lambda rows: [{**x, "success": "0"} if x["probe"] == "P0_success" else x
                       for x in rows]),
     "really is a success state"),

    ("the grasped probe was never measured (stale or missing fixture)",
     lambda o, r, t: _edit_csv(
         os.path.join(o, f"probes_{MODE}.csv"),
         lambda rows: [x for x in rows if x["probe"] != "P7_held"]),
     "P0_success > P7_held"),

    ("the reward rises as the cube moves away from the target",
     lambda o, r, t: _edit_csv(
         os.path.join(o, f"sweeps_{MODE}.csv"),
         lambda rows: [{**x, "reward": 0.3 * int(x["i"])} for x in rows]),
     "does not rise as the cube moves away"),

    ("the reward ignores height entirely - flat along the z sweep",
     lambda o, r, t: _edit_csv(
         os.path.join(o, f"sweeps_{MODE}.csv"),
         lambda rows: [{**x, "reward": 4.0} if x["sweep"] == "z" else x
                       for x in rows]),
     "actually varies along this axis"),

    ("cumulative reward carries no information about the outcome",
     lambda o, r, t: _align(o, _shuffle_returns), "ranks successes above failures"),

    ("a grasp-farming reward: pays for holding, not for placing",
     lambda o, r, t: _align(o, _grasp_farm), "rises with the stage reached"),

    ("the jitter policy accumulates more reward than the real one",
     lambda o, r, t: _align(o, lambda rows: [
         {**x, "ep_return": 5000.0} if x["arm"] == "jitter" else x
         for x in rows]), "beats `jitter`"),

    ("doing nothing accumulates more reward than acting",
     lambda o, r, t: _align(o, lambda rows: [
         {**x, "ep_return": 5000.0} if x["arm"] == "zero" else x
         for x in rows]), "beats `zero`"),

    ("an initial state disagrees with T-II's record of that seed",
     lambda o, r, t: _align(o, lambda rows: [
         {**x, "cubeA_x": 0.1234} if i == 3 else x
         for i, x in enumerate(rows)]), "match T-II's record"),

    ("the biased sampler was left ON during the measurement",
     lambda o, r, t: _edit_json(os.path.join(o, f"align_{MODE}_manifest.json"),
                                lambda j: {**j, "t3_sampler": 1}),
     "biased sampler was OFF"),

    ("REWARD_MAX in the manifest is not the file's",
     lambda o, r, t: _edit_json(os.path.join(o, f"align_{MODE}_manifest.json"),
                                lambda j: {**j, "reward_max": 3.0}),
     "REWARD_MAX matches"),

    ("the run is truncated - a crashed job, not a finished one",
     lambda o, r, t: _align(o, lambda rows: rows[:20]), "enough episodes"),

    ("the sampler barely reaches the target region",
     lambda o, r, t: _edit_json(os.path.join(o, f"sampler_{MODE}.json"),
                                lambda j: {**j, "hit_rate": 0.05,
                                           "enrichment": 1.1}),
     "land in the failure region"),

    ("the sampler has collapsed onto one configuration",
     lambda o, r, t: _edit_json(os.path.join(o, f"sampler_{MODE}.json"),
                                lambda j: {**j, "sd_cubeA_x": 0.0,
                                           "distinct_frac": 0.01}),
     "spread"),

    ("the sampler is not reproducible from a torch seed",
     lambda o, r, t: _edit_json(os.path.join(o, f"sampler_{MODE}.json"),
                                lambda j: {**j, "deterministic": False}),
     "same torch seed"),

    ("the reward mutates the simulator while scoring it",
     lambda o, r, t: _edit_json(
         os.path.join(o, f"probes_{MODE}.json"),
         lambda j: {**j, "layer_b": {**j["layer_b"], "mutation_ok": False}}),
     "does not mutate"),

    ("the reward is not a pure function of the state",
     lambda o, r, t: _edit_json(
         os.path.join(o, f"probes_{MODE}.json"),
         lambda j: {**j, "layer_b": {**j["layer_b"], "pure_ok": False}}),
     "pure"),

    ("the generated reward violates the contract",
     lambda o, r, t: shutil.copyfile(
         os.path.join(HERE, "fixtures", "bad_import_os.py"),
         os.path.join(r, "reward.py")), "1 static"),
]


def _shuffle_returns(rows):
    """Same returns, reassigned at random - the reward carries no signal."""
    pol = [r for r in rows if r["arm"] == "policy"]
    rets = [r["ep_return"] for r in pol]
    random.Random(3).shuffle(rets)
    for r, v in zip(pol, rets):
        r["ep_return"] = v
    return rows


def _grasp_farm(rows):
    """Return depends on grasping and nothing else, plus a success bonus.

    The commonest hack there is, and the reason the stage-margin check exists:
    'grasped but never placed' ends up scoring ABOVE 'placed but it fell off',
    which no success bonus hides.
    """
    for r in rows:
        if r["arm"] != "policy":
            continue
        g = r["ever_grasped"] == "1"
        r["ep_return"] = (1000.0 if r["success_once"] == "1"
                          else (900.0 if g and r["ever_placed"] != "1"
                                else (300.0 if g else 100.0)))
    return rows


def main():
    root = tempfile.mkdtemp(prefix="t3verify")
    try:
        out, run, t2 = build(root)
        rc, log = run_gate(out, run, t2)
        check(rc == 0, "a valid pass is accepted",
              "exit 0" if rc == 0 else
              "\n".join(l for l in log.splitlines() if "FAIL" in l))
        if rc != 0:
            print("\n  The fabricated pass does not itself pass. Every "
                  "corruption below\n  would 'fail' for the wrong reason, so "
                  "stopping here.")
            sys.exit(1)

        print(f"\n  and each of {len(CORRUPTIONS)} corruptions is caught\n")
        for desc, corrupt, want in CORRUPTIONS:
            sub = tempfile.mkdtemp(prefix="t3c", dir=root)
            o2, r2, t22 = build(sub)
            corrupt(o2, r2, t22)
            rc, log = run_gate(o2, r2, t22)
            if rc == 0:
                check(False, desc, "the gate ACCEPTED it")
                continue
            hit = [l.strip() for l in log.splitlines()
                   if l.strip().startswith("FAIL") and want in l]
            check(bool(hit), desc,
                  hit[0][:110] if hit else
                  f"rejected, but not for {want!r}: "
                  + "; ".join(l.strip()[:60] for l in log.splitlines()
                              if l.strip().startswith("FAIL"))[:200])
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print()
    if FAILS:
        print(f"{len(FAILS)} FAILED: {'; '.join(FAILS)}")
        sys.exit(1)
    print(f"the gate accepts a valid pass and rejects all "
          f"{len(CORRUPTIONS)} corruptions, each for the right reason.")


if __name__ == "__main__":
    main()
