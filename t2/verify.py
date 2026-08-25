#!/usr/bin/env python
"""Check that a finished T-II pass is what it claims to be, before it is reported.

Every assertion here corresponds to a way one of these numbers could be quietly
wrong, and most of them have been wrong at least once in this repo:

  * the T-I pass ran on seeds 0-299, which are DEMONSTRATION seeds, and reported
    memorisation (0.910) as a success rate (0.713);
  * a "which cube is far" split turned out to measure "how many cubes are far";
  * region episodes sharing seeds with the pass that found the region would
    measure noise rather than a failure mode.

It reads only the committed CSVs. No sim, no GPU, no checkpoint.

  python t2/verify.py $RIPL_ROOT/t2

Exit status is 0 only if every check passes, so it can gate a report.
"""
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t2_common import PREDICTED  # noqa: E402
from t2_common import REGIONS as SHARED_REGIONS  # noqa: E402
from t2_common import geom_from_row, wilson  # noqa: E402

DEMO_SEED_CEILING = 1000
FAILS = []


def check(ok, label, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILS.append(label)
    return ok


# A rollout CSV is identified by its SCHEMA, not by its filename. The output
# directory also holds the seed index, the per-mode seed lists and
# demo_feasibility.csv, and the last of those carries a `seed` column - so a
# filename-exclusion list silently let it through and every check that assumed
# cube columns then read a rollout that was not one. Ask for the columns the
# checks actually need instead.
ROLLOUT_COLS = ("seed", "run_id", "cubeA_x", "cubeA_y", "success_once")


def rollouts(out):
    """(basename, rows) for every rollout CSV in `out`, sorted."""
    found = []
    for f in sorted(glob.glob(f"{out}/*.csv")):
        rows = list(csv.DictReader(open(f)))
        if rows and all(c in rows[0] for c in ROLLOUT_COLS):
            found.append((os.path.basename(f)[:-4], rows))
    return found


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r.update(geom_from_row(r))
    return rows


# The filters each region pass claims to have applied, and the nominal-pass
# prediction each was meant to confirm - both imported, so this file and the
# miner cannot disagree about what "the gap region" means. The legacy
# separation bands are kept so the superseded passes still verify.
REGIONS = dict(SHARED_REGIONS)
REGIONS["near"] = lambda g: g["separation"] < 0.080
REGIONS["far"] = lambda g: g["separation"] >= 0.260


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "t2/results"
    print(f"verifying {out}\n")

    # ---- 1. nothing anywhere evaluates on the training seeds ---------------
    print("1. no evaluation block touches the demonstration seeds [0, 1000)")
    for b, rows in rollouts(out):
        seeds = [int(float(r["seed"])) for r in rows if r.get("seed")]
        if not seeds:
            continue
        low = [s for s in seeds if s < DEMO_SEED_CEILING]
        # t1_trainseed* is the memorisation measurement and is SUPPOSED to be
        # in there. Anything else is the bug that produced 0.910.
        expected = "trainseed" in b
        check(bool(low) == expected, f"{b}",
              f"min seed {min(seeds)}, {len(low)} below {DEMO_SEED_CEILING}"
              + ("  (intentional: memorisation control)" if expected else ""))

    # ---- 2. the passes do not share episodes ------------------------------
    print("\n2. rollout passes use disjoint seed blocks")
    blocks = {}
    for b, rows in rollouts(out):
        if "trainseed" in b:
            continue          # deliberately overlaps the demo seeds; check 1 owns it
        sd = set(int(float(r["seed"])) for r in rows if r.get("seed"))
        if sd:
            blocks[b] = sd
    names = sorted(blocks)
    for i, a in enumerate(names):
        for c in names[i + 1:]:
            # blocks of the same region pass are meant to be disjoint too
            ov = blocks[a] & blocks[c]
            check(not ov, f"{a} vs {c}", f"{len(ov)} shared seeds" if ov else "")

    # ---- 3. every region episode actually satisfies its filter -------------
    print("\n3. region episodes satisfy the filter their pass claims")
    for tag, pred in REGIONS.items():
        # Exact tag match only. A prefix glob on 'near' would also swallow
        # every 'nearbase' file and test them against the wrong filter, which
        # would pass silently for the ones that happen to satisfy both.
        files = (glob.glob(f"{out}/region_{tag}.csv")
                 + glob.glob(f"{out}/region_{tag}_seed[0-9].csv"))
        if not files:
            continue
        rows = [r for f in files for r in load(f)]
        for r in rows:
            r["separation"] = float(r["separation"])
        bad = [r for r in rows if not pred(r)]
        check(not bad, f"region '{tag}' ({len(rows)} episodes)",
              f"{len(bad)} episodes outside the region" if bad else "all inside")

    # ---- 3b. the episodes ARE the seeds that were asked for ---------------
    #
    # The strongest offline check available, and it is free. The cube poses in
    # a rollout CSV were read back out of the ENVIRONMENT at reset, through
    # CubePoseInfo. The cube poses in seeds.csv were produced by seed_index.py,
    # a separate policy-free script that only calls reset(). If they agree for
    # every episode then: the env really did reset to the seed that was
    # requested, the seed -> state map is genuine and deterministic, and the
    # filter selected seeds whose ACTUAL states satisfy the condition. Three
    # claims, one join, no simulator.
    print("\n3b. logged initial states match the independent seed index")
    idx_path = f"{out}/seeds.csv"
    if not os.path.exists(idx_path):
        check(False, "seeds.csv present", "cannot cross-check without the index")
    else:
        idx = {int(r["seed"]): r for r in csv.DictReader(open(idx_path))}
        for b, rows in rollouts(out):
            checked = mism = 0
            worst = 0.0
            missing = 0
            for r in rows:
                sd = int(float(r["seed"]))
                if sd not in idx:
                    missing += 1
                    continue
                checked += 1
                for k in ("cubeA_x", "cubeA_y", "cubeA_theta",
                          "cubeB_x", "cubeB_y", "cubeB_theta"):
                    d = abs(float(r[k]) - float(idx[sd][k]))
                    worst = max(worst, d)
                    if d > 1e-5:
                        mism += 1
                        break
            detail = f"{checked} episodes, max coord diff {worst:.2e}"
            if missing:
                detail += f", {missing} seeds not in the index"
            check(mism == 0 and checked > 0, b, detail)

    # ---- 3c. one checkpoint produced all of it ----------------------------
    print("\n3c. every rollout pass used the same checkpoint")
    shas = {}
    for f in sorted(glob.glob(f"{out}/*_manifest.json")):
        import json
        m = json.load(open(f))
        if "ckpt_sha256" in m:
            shas.setdefault(m["ckpt_sha256"], []).append(os.path.basename(f))
    if not shas:
        print("     (no ckpt_sha256 in any manifest - runs predate the hash;"
              " re-run to get attribution)")
    else:
        for h, files in shas.items():
            print(f"     {h}  {len(files)} pass(es)")
        check(len(shas) == 1, "single checkpoint across all passes",
              f"{len(shas)} distinct checkpoints" if len(shas) > 1 else "")

    # ---- 4. the deliverable's shape: 100 rollouts x 3 seeds ---------------
    print("\n4. per-mode evaluation is 100 rollouts x 3 seeds")
    for tag in REGIONS:
        files = sorted(f for f in glob.glob(f"{out}/region_{tag}_seed[0-9].csv"))
        if not files:
            continue
        ns, seeds_used = [], set()
        for f in files:
            rows = list(csv.DictReader(open(f)))
            ns.append(len(rows))
            seeds_used |= {int(float(r["policy_seed"])) for r in rows}
        check(len(files) == 3 and all(n == 100 for n in ns) and len(seeds_used) == 3,
              f"region '{tag}'",
              f"{len(files)} blocks, sizes {ns}, policy seeds {sorted(seeds_used)}")

    # ---- 5. the numbers, with the prediction each was meant to confirm -----
    print("\n5. per-mode success rate (mean +- SD over the 3 policy seeds)")
    print(f"   {'mode':>10} {'per-seed rates':>26} {'mean':>7} {'SD':>7} "
          f"{'pooled 95% CI':>16}  predicted")
    for tag in list(REGIONS) + ["nominal"]:
        pat = (f"{out}/region_{tag}_seed[0-9].csv" if tag != "nominal"
               else f"{out}/nominal.csv")
        files = sorted(glob.glob(pat))
        if not files:
            continue
        rates, k, n = [], 0, 0
        for f in files:
            rows = list(csv.DictReader(open(f)))
            ki = sum(1 for r in rows if r.get("success_once") == "1"
                     or r.get("ever_success") == "1")
            rates.append(ki / len(rows)); k += ki; n += len(rows)
        m = sum(rates) / len(rates)
        sd = (sum((x - m) ** 2 for x in rates) / (len(rates) - 1)) ** 0.5 \
            if len(rates) > 1 else float("nan")
        lo, hi = wilson(k, n)
        p = PREDICTED.get(tag)
        pred = f"{p[0]:.3f} [{p[1]:.3f}, {p[2]:.3f}]" if p else ""
        flag = ""
        if p:
            flag = "  <- OUTSIDE predicted CI" if not (p[1] <= m <= p[2]) else "  ok"
        print(f"   {tag:>10} {' '.join(f'{x:.3f}' for x in rates):>26} "
              f"{m:7.3f} {sd:7.3f} [{lo:.3f}, {hi:.3f}]  {pred}{flag}")

    print("")
    if FAILS:
        print(f"{len(FAILS)} CHECK(S) FAILED - do not report these numbers:")
        for f in FAILS:
            print(f"  - {f}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
