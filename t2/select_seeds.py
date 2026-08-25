#!/usr/bin/env python
"""Pick seeds out of the index by an arbitrary condition on the initial state.

Replaces the separation-only filter that used to live inline in run_t2.sh. The
refined failure modes are regions in face_gap and in distance from the Panda's
base, not in centre separation, so the selector has to speak those axes too --
and the next mode after these will want an axis nobody has thought of yet.

The seed index is what makes this cheap: reset(seed=s) is deterministic, so the
index is a lossless table of seed -> initial state and "resample fresh episodes
from the failure region" is rejection sampling over integers. No state
injection, no distribution shift, every episode reproducible from one integer.

Usage:
  python t2/select_seeds.py seeds.csv --min-seed 2200 -n 200 \
         --where "face_gap < 0.025 and 0.52 <= dist_min and dist_max < 0.76" \
         --out region_gap_seeds.csv

  --dry-run prints the hit rate and stops, which is how a pass gets sized
  before it is launched rather than after it runs short.
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t2_common import GEOM_FIELDS, geom_from_row  # noqa: E402

# What --where may reference. Everything here is a property of the INITIAL
# state, which is the point: a T-II failure mode is a region of the
# initial-state distribution, so a filter that could see an outcome column
# would let a "region" be defined by the thing it is supposed to predict.
FEATURES = ("seed", "separation", "relative_yaw", "relative_yaw_mod90",
            "cubeA_x", "cubeA_y", "cubeA_theta",
            "cubeB_x", "cubeB_y", "cubeB_theta") + GEOM_FIELDS


def features(row):
    f = {k: float(row[k]) for k in FEATURES if k in row and k not in GEOM_FIELDS}
    f.update(geom_from_row(row))
    f["seed"] = int(row["seed"])
    return f


def main():
    p = argparse.ArgumentParser()
    p.add_argument("index", help="seeds.csv from seed_index.py")
    p.add_argument("--where", default="True",
                   help="python expression over: " + ", ".join(FEATURES))
    p.add_argument("--min-seed", type=int, default=0,
                   help="floor, to keep passes disjoint from each other and "
                        "from the demonstration seeds")
    p.add_argument("-n", "--num", type=int, default=200)
    p.add_argument("--out")
    p.add_argument("--split", type=int, default=0,
                   help="also write N disjoint blocks as <out>_1.csv ... "
                        "<out>_N.csv, for an M-rollouts x N-seeds evaluation")
    p.add_argument("--dry-run", action="store_true")
    # sugar, so the separation-band invocations already written down elsewhere
    # keep working unchanged
    p.add_argument("--min-sep", type=float)
    p.add_argument("--max-sep", type=float)
    a = p.parse_args()

    where = a.where
    if a.min_sep is not None or a.max_sep is not None:
        lo = a.min_sep if a.min_sep is not None else 0.0
        hi = a.max_sep if a.max_sep is not None else 9.0
        where = f"({where}) and {lo} <= separation < {hi}"

    rows = list(csv.DictReader(open(a.index)))
    if not rows:
        sys.exit(f"!! {a.index} is empty")
    eligible = [r for r in rows if int(r["seed"]) >= a.min_seed]

    code = compile(where, "<--where>", "eval")
    hits = []
    for r in eligible:
        f = features(r)
        try:
            if eval(code, {"__builtins__": {}}, f):
                hits.append((r, f))
        except NameError as e:
            sys.exit(f"!! --where references something that is not a feature: {e}\n"
                     f"   available: {', '.join(FEATURES)}")

    rate = len(hits) / len(eligible) if eligible else 0.0
    print(f"  index      {len(rows)} seeds, {len(eligible)} at or above seed {a.min_seed}")
    print(f"  where      {where}")
    print(f"  hits       {len(hits)}  ({rate:.2%} of eligible)")
    if rate:
        # What the index would have to hold for this pass to fill from a given
        # floor. CLAUDE.md's rule: size a run before launching it, not after it
        # comes up short at hour three.
        print(f"  to get {a.num}: about {a.min_seed + int(a.num / rate):,} indexed seeds")

    if hits:
        for k in ("separation", "face_gap", "dist_min", "dist_max"):
            v = sorted(f[k] for _, f in hits)
            print(f"  {k:11s} {v[0] * 1000:7.1f} .. {v[len(v) // 2] * 1000:7.1f} "
                  f".. {v[-1] * 1000:7.1f} mm  (min/median/max)")

    if a.dry_run:
        return
    if len(hits) < a.num:
        sys.exit(f"\n!! only {len(hits)} seeds match, wanted {a.num}.\n"
                 f"   Raise INDEX_SEEDS, lower --min-seed, or widen the band.\n")
    if not a.out:
        sys.exit("!! --out is required unless --dry-run")

    chosen = hits[:a.num]
    # Write the geometry alongside, so the seed list records the region it was
    # drawn from and a later reader does not have to recompute it to find out.
    fields = list(chosen[0][0].keys()) + [k for k in GEOM_FIELDS
                                          if k not in chosen[0][0]]

    def dump(path, sel):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r, feat in sel:
                w.writerow({**r, **{k: feat[k] for k in GEOM_FIELDS}})

    dump(a.out, chosen)
    print(f"\nwrote {a.out}  ({len(chosen)} seeds, all >= {a.min_seed})")

    if a.split > 1:
        # DISJOINT blocks, not repeats of one block. Each gets its own policy
        # seed downstream, so the three success rates are independent estimates
        # and their spread is a real error bar. Reusing one block and varying
        # only the policy seed would hold the initial states fixed and measure
        # DDPM sampling noise instead - a much smaller quantity, and not what
        # "3 seeds" means. Same argument as run_t2.sh's do_t1.
        if len(chosen) % a.split:
            sys.exit(f"!! {len(chosen)} seeds do not divide into {a.split} "
                     f"equal blocks; ask for a multiple of {a.split}.")
        per = len(chosen) // a.split
        base = os.path.splitext(a.out)[0]
        for i in range(a.split):
            path = f"{base}_{i + 1}.csv"
            dump(path, chosen[i * per:(i + 1) * per])
            print(f"       {path}  ({per} seeds)")


if __name__ == "__main__":
    main()
