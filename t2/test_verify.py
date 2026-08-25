#!/usr/bin/env python
"""Check that t2/verify.py actually catches things. No deps, no pod, no GPU.

    python3 t2/test_verify.py

A checker nobody has tried to fool is not evidence. This fabricates a valid
evaluation output - the exact shape eval_modes.py writes - confirms verify.py
passes it, then corrupts it twelve ways, one at a time, and confirms verify.py
fails each one and says why.

The twelve corruptions are the ways these numbers could be quietly wrong. Two
of them have actually happened in this repo (evaluating on demonstration seeds;
a region pass reaching into the T-I block), and one is the misreading of "3
seeds" that the harness shipped with until it was checked.

Runs in about two seconds on the bare system interpreter, because everything it
touches - geometry.py and verify.py - imports nothing but the standard library.
"""
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from geometry import (COLUMNS, EVAL_BASE, MODES, cube_features,  # noqa: E402
                      reserved_hit)

VERIFY = os.path.join(HERE, "verify.py")


# ---------------------------------------------------------------------------
# fabricate a valid pass
# ---------------------------------------------------------------------------


def quat_z(t):
    return [math.cos(t / 2), 0.0, 0.0, math.sin(t / 2)]


def draw(rng):
    """One initial state, drawn the way StackCube's _initialize_episode does:
    a shared xy offset for the pair, then a rejection-sampled placement each,
    with the sampler's 58.6 mm centre-separation floor."""
    ox, oy = rng.uniform(-0.1, 0.1), rng.uniform(-0.2, 0.2)
    while True:
        ax, ay = ox + rng.uniform(-0.1, 0.1), oy + rng.uniform(-0.1, 0.1)
        bx, by = ox + rng.uniform(-0.1, 0.1), oy + rng.uniform(-0.1, 0.1)
        if math.dist((ax, ay), (bx, by)) >= 0.0586:
            return ([ax, ay, 0.02] + quat_z(rng.uniform(0, 2 * math.pi)),
                    [bx, by, 0.02] + quat_z(rng.uniform(0, 2 * math.pi)))


def fabricate(out, modes=("nominal", "gap", "farb"), per_block=100, blocks=3):
    """Write a seed index and 3 blocks per mode, exactly as eval_modes.py does."""
    rng = random.Random(7)
    want = per_block * blocks
    picked = {m: [] for m in modes}
    index, used, seed = [], set(), EVAL_BASE

    while any(len(picked[m]) < want for m in modes):
        seed += 1
        if reserved_hit(seed):
            continue
        f = cube_features(*draw(rng))
        index.append(dict(seed=seed, **f))
        # Fixed order, one shared shrinking pool - the same allocation
        # eval_modes.select_seeds does, and the reason blocks are disjoint.
        for tag in ("gap", "farb", "nominal"):
            if tag in picked and len(picked[tag]) < want \
                    and MODES[tag](f) and seed not in used:
                picked[tag].append(seed)
                used.add(seed)
                break

    with open(f"{out}/seeds.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(index[0].keys()))
        w.writeheader()
        w.writerows(index)
    by_seed = {r["seed"]: r for r in index}

    # Outcome rates from the 1,200-episode discovery pass, so the fabricated
    # table lands where the real one should and the DISCOVERY comparison in
    # verify.py's section 7 is exercised rather than trivially skipped.
    rate = {"nominal": 0.713, "gap": 0.523, "farb": 0.561}
    for tag in modes:
        for b in range(1, blocks + 1):
            rows = []
            for s in picked[tag][(b - 1) * per_block:b * per_block]:
                f = by_seed[s]
                succ = rng.random() < rate[tag]
                rows.append({**{k: f.get(k, "") for k in COLUMNS},
                             "run_id": f"mode_{tag}_seed{b}", "mode": tag,
                             "block": b, "policy_seed": b, "seed": s,
                             "success_once": int(succ), "success_at_end": int(succ),
                             "ep_len": 200, "ever_grasped": 1,
                             "ever_placed": int(succ or rng.random() < 0.4),
                             "ever_static": int(succ),
                             "final_cubeA_x": 0.0, "final_cubeA_y": 0.0,
                             "final_cubeA_z": 0.02, "cubeB_displacement": 0.001})
            with open(f"{out}/mode_{tag}_seed{b}.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=COLUMNS)
                w.writeheader()
                w.writerows(rows)
            json.dump({"ckpt_sha256": "deadbeefcafe1234", "mode": tag, "block": b},
                      open(f"{out}/mode_{tag}_seed{b}_manifest.json", "w"))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def run_verify(d):
    p = subprocess.run([sys.executable, VERIFY, d], capture_output=True, text=True)
    fails = [ln.strip()[6:] for ln in p.stdout.splitlines()
             if ln.strip().startswith("FAIL")]
    return p.returncode, fails, p.stdout


def drop_column(path, col):
    """Remove a column from the header AND every row - a genuine schema drift,
    as opposed to leaving the column present and empty (which is a different
    corruption, and is tested separately)."""
    rows = list(csv.DictReader(open(path)))
    hdr = [c for c in rows[0] if c != col]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def edit_csv(path, fn):
    rows = list(csv.DictReader(open(path)))
    hdr = list(rows[0].keys())
    fn(rows)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    root = tempfile.mkdtemp(prefix="t2-verify-test-")
    good = os.path.join(root, "good")
    os.makedirs(good)
    print("fabricating a valid pass (9 blocks x 100 episodes)...")
    fabricate(good)

    ok = True

    # --- the valid pass must PASS ------------------------------------------
    rc, fails, stdout = run_verify(good)
    if rc == 0:
        print("  PASS   a valid pass verifies clean")
    else:
        ok = False
        print(f"  FAIL   a valid pass was rejected: {fails}")
        print(stdout)

    # --- and each corruption must FAIL -------------------------------------
    # (label, what it simulates, mutation)
    cases = [
        ("seed inside the demonstration range",
         "the bug that reported 0.910 memorisation as a success rate",
         lambda d: edit_csv(f"{d}/mode_gap_seed1.csv",
                            lambda r: r[0].__setitem__("seed", "42"))),
        ("two blocks share a seed",
         "correlated blocks, so the SD understates the error bar",
         lambda d: edit_csv(f"{d}/mode_gap_seed2.csv", lambda r: r[0].update(
             next(iter(csv.DictReader(open(f"{d}/mode_gap_seed1.csv"))))))),
        ("an episode is outside its mode's region",
         "a selection bug filing episodes under the wrong mode",
         lambda d: edit_csv(f"{d}/mode_gap_seed1.csv", lambda r: r[0].update(
             cubeA_x="0.0", cubeA_y="0.0", cubeB_x="0.30", cubeB_y="0.0"))),
        ("a geometry column was doctored",
         "stored features that disagree with the poses they came from",
         lambda d: edit_csv(f"{d}/mode_gap_seed1.csv",
                            lambda r: r[0].__setitem__("face_gap", "0.001"))),
        ("the seed index disagrees with the rollout",
         "the env not resetting to the seed that was requested",
         lambda d: edit_csv(f"{d}/seeds.csv",
                            lambda r: r[0].__setitem__("cubeA_x", "0.987"))),
        ("two different checkpoints across the blocks",
         "mixing the pooled-encoder arm into the spatial one's table",
         lambda d: json.dump({"ckpt_sha256": "0" * 16},
                             open(f"{d}/mode_gap_seed1_manifest.json", "w"))),
        ("no manifests at all",
         "numbers with no attribution to the weights that produced them",
         lambda d: [os.remove(os.path.join(d, f)) for f in os.listdir(d)
                    if f.endswith("_manifest.json")]),
        ("a block is short of 100 episodes",
         "a pass that ran out of seeds and quietly under-filled",
         lambda d: edit_csv(f"{d}/mode_gap_seed1.csv", lambda r: r.pop())),
        ("one block run 3x instead of 3 disjoint blocks",
         "the misreading of '3 seeds' that measures DDPM noise, not variance",
         lambda d: [edit_csv(f"{d}/mode_gap_seed{b}.csv",
                             lambda r: [x.update(policy_seed="1") for x in r])
                    for b in (1, 2, 3)]),
        ("a column is missing from the CSV",
         "a schema drift between the writer and the readers",
         lambda d: drop_column(f"{d}/mode_gap_seed1.csv", "face_gap")),
        ("an outcome column is blank",
         "a broken info path reading as a badly performing policy",
         lambda d: edit_csv(f"{d}/mode_farb_seed2.csv",
                            lambda r: r[7].__setitem__("success_once", ""))),
        ("a cell is blank",
         "a CSV half-written by an interrupted run",
         lambda d: edit_csv(f"{d}/mode_gap_seed1.csv",
                            lambda r: r[3].__setitem__("face_gap", ""))),
    ]

    print(f"\ncorrupting it {len(cases)} ways, one at a time:")
    for label, why, mutate in cases:
        d = os.path.join(root, label.replace(" ", "_")[:40])
        shutil.rmtree(d, ignore_errors=True)
        shutil.copytree(good, d)
        mutate(d)
        rc, fails, _ = run_verify(d)
        if rc != 0:
            print(f"  PASS   caught: {label}")
            print(f"           -> {fails[0][:96] if fails else '(non-zero exit, no FAIL line)'}")
        else:
            ok = False
            print(f"  FAIL   MISSED: {label}")
            print(f"           ({why})")

    shutil.rmtree(root, ignore_errors=True)
    print("")
    if ok:
        print("verify.py passes valid data and catches every corruption")
        return 0
    print("verify.py has a hole - do not rely on it to gate a report")
    return 1


if __name__ == "__main__":
    sys.exit(main())
