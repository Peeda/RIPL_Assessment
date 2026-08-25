#!/usr/bin/env python
"""Assert a finished evaluation is what it claims, before anyone reports it.

    python t2/verify.py $RIPL_ROOT/t2

Reads only the committed CSVs and manifests. No simulator, no GPU, no
checkpoint, no numpy - so it runs on a laptop, in a second, against the evidence
that will actually be quoted.

Exit status is 0 only if every check passes, so it can gate a report.

eval_modes.py already refuses to LOG an episode that fails its reset-time
assertions. This is the independent re-check: it re-derives the geometry from
the logged columns rather than trusting the ones eval_modes computed, and it
checks the properties that are only visible ACROSS files - disjointness between
blocks, one checkpoint behind every pass, the 100 x 3 shape.

Every check here corresponds to a way one of these numbers has been, or could
quietly be, wrong:

  * the first T-I pass ran on seeds 0-299, which are DEMONSTRATION seeds, and
    reported memorisation (0.910) as a success rate (0.713);
  * the old region selector reached through T-I's [6000, 6300) block and pulled
    20 of its seeds into a "fresh" region pass;
  * a region pass re-measured on the episodes that identified the region would
    measure noise rather than a failure mode.
"""
import csv
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry import (COLUMNS, DISCOVERY, MODES, RESERVED,  # noqa: E402
                      geom_from_row, reserved_hit, wilson)

FAILS = []


def check(ok, label, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILS.append(label)
    return ok


def blocks(out):
    """[(mode, block, path, rows)] for every per-mode evaluation block."""
    found = []
    for f in sorted(glob.glob(f"{out}/mode_*_seed[0-9].csv")):
        m = re.match(r"mode_(.+)_seed(\d)\.csv$", os.path.basename(f))
        if not m:
            continue
        rows = list(csv.DictReader(open(f)))
        if rows:
            found.append((m.group(1), int(m.group(2)), f, rows))
    return found


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "t2/results"
    print(f"verifying {out}\n")

    found = blocks(out)
    if not found:
        print(f"!! no mode_<tag>_seed<b>.csv under {out}\n"
              f"   Run: python t2/eval_modes.py CKPT --out {out}")
        sys.exit(1)

    # ---- 1. nothing evaluates on a reserved block -------------------------
    print("1. no evaluation block touches a reserved seed range")
    for name, (lo, hi) in RESERVED.items():
        print(f"     reserved '{name}': [{lo}, {hi})")
    for tag, b, _, rows in found:
        seeds = [int(r["seed"]) for r in rows]
        bad = {s: reserved_hit(s) for s in seeds if reserved_hit(s)}
        check(not bad, f"mode '{tag}' block {b}",
              f"min seed {min(seeds)}, max {max(seeds)}"
              + (f", {len(bad)} in reserved blocks" if bad else ""))

    # ---- 1b. the CSV is the schema geometry.COLUMNS declares --------------
    #
    # FATAL rather than merely recorded, and checked before anything else reads
    # a column: every check below indexes by name, so a schema violation does
    # not produce one finding among many - it invalidates the rest of the run.
    # Reporting it and stopping beats a KeyError traceback three checks later.
    print("\n1b. every block's header is exactly geometry.COLUMNS")
    schema_ok = True
    for tag, b, path, rows in found:
        got = list(rows[0].keys())
        schema_ok &= check(got == COLUMNS, f"mode '{tag}' block {b}",
                           "" if got == COLUMNS else
                           f"missing {[c for c in COLUMNS if c not in got]}, "
                           f"extra {[c for c in got if c not in COLUMNS]}")
    if not schema_ok:
        print("\n  The CSV schema does not match geometry.COLUMNS, so every check"
              "\n  below would be reading columns that may not mean what they say."
              "\n  Stopping here. Re-run eval_modes.py against the current schema.")
        print(f"\n{len(FAILS)} CHECK(S) FAILED - do not report these numbers:")
        for f in FAILS:
            print(f"  - {f}")
        sys.exit(1)

    # ---- 2. every block is disjoint from every other ----------------------
    #
    # Across modes AND within a mode. Two blocks sharing an episode would make
    # their rates correlated, and the SD over the three would understate the
    # error bar it is there to provide.
    print("\n2. all blocks use disjoint seed sets")
    sets = {(tag, b): set(int(r["seed"]) for r in rows) for tag, b, _, rows in found}
    keys = sorted(sets)
    for i, a in enumerate(keys):
        for c in keys[i + 1:]:
            ov = sets[a] & sets[c]
            check(not ov, f"{a[0]}#{a[1]} vs {c[0]}#{c[1]}",
                  f"{len(ov)} shared seeds" if ov else "")

    # ---- 3. every episode satisfies the mode it is filed under ------------
    #
    # Recomputed from the logged cube poses, not read off the stored geometry
    # columns - so a bug in the geometry eval_modes wrote would be caught here
    # rather than confirmed.
    print("\n3. every episode's initial state satisfies its mode's filter")
    for tag, b, _, rows in found:
        if tag not in MODES:
            check(False, f"mode '{tag}'", "no such mode in geometry.MODES")
            continue
        bare = [{k: r[k] for k in ("cubeA_x", "cubeA_y", "cubeA_theta",
                                   "cubeB_x", "cubeB_y", "cubeB_theta")}
                for r in rows]
        bad = [i for i, g in enumerate(bare) if not MODES[tag](geom_from_row(g))]
        check(not bad, f"mode '{tag}' block {b} ({len(rows)} episodes)",
              f"{len(bad)} outside the region" if bad else "all inside")

    # ---- 3b. the stored geometry columns agree with a fresh recompute ------
    #
    # A cell that will not parse - blank, truncated, garbage - is a finding in
    # its own right and is reported as one. A partially written CSV from an
    # interrupted run is the realistic way this happens, and it must not come
    # back as a ValueError traceback that looks like a bug in the checker.
    print("\n3b. stored geometry columns match a recompute from the poses")
    for tag, b, _, rows in found:
        worst, key, unparseable = 0.0, "", []
        for i, r in enumerate(rows):
            fresh = geom_from_row({k: r[k] for k in ("cubeA_x", "cubeA_y",
                                                     "cubeA_theta", "cubeB_x",
                                                     "cubeB_y", "cubeB_theta")})
            for k, v in fresh.items():
                raw = r.get(k)
                try:
                    got = float(raw)
                except (TypeError, ValueError):
                    unparseable.append((i, k, raw))
                    continue
                d = abs(v - got)
                if d > worst:
                    worst, key = d, k
        if unparseable:
            i, k, raw = unparseable[0]
            check(False, f"mode '{tag}' block {b}",
                  f"{len(unparseable)} cell(s) will not parse as a number, "
                  f"first: row {i} column '{k}' = {raw!r}")
        else:
            check(worst < 1e-9, f"mode '{tag}' block {b}",
                  f"max diff {worst:.2e}" + (f" in {key}" if worst else ""))

    # ---- 3c. no outcome column is blank ------------------------------------
    #
    # A blank success_once is not '0' - it means the metric never arrived. But
    # every rate below compares against '1', so a blank counts as a failure and
    # a broken info path reads as a badly performing policy. Name it instead.
    print("\n3c. no outcome column is blank")
    for tag, b, _, rows in found:
        cols = ("success_once", "success_at_end", "ever_grasped",
                "ever_placed", "ever_static")
        blank = [(i, c) for i, r in enumerate(rows) for c in cols
                 if str(r[c]).strip() == ""]
        check(not blank, f"mode '{tag}' block {b}",
              f"{len(blank)} blank cell(s), first: row {blank[0][0]} "
              f"column '{blank[0][1]}'" if blank else "")

    # ---- 4. the episodes ARE the seeds that were asked for -----------------
    #
    # The strongest offline check available, and it is free. The cube poses in
    # an evaluation CSV were read back out of the ENVIRONMENT at reset, through
    # CubePoseInfo. The cube poses in seeds.csv were produced by seed_index.py,
    # a separate policy-free script that only calls reset(). If they agree for
    # every episode then: the env really did reset to the seed requested, the
    # seed -> state map is genuine and deterministic, and the filter selected
    # seeds whose ACTUAL states satisfy the condition. Three claims, one join,
    # no simulator.
    print("\n4. logged initial states match the independent seed index")
    idx_path = f"{out}/seeds.csv"
    if not os.path.exists(idx_path):
        check(False, "seeds.csv present", "cannot cross-check without the index")
    else:
        idx = {int(r["seed"]): r for r in csv.DictReader(open(idx_path))}
        for tag, b, _, rows in found:
            checked = mism = missing = 0
            worst = 0.0
            for r in rows:
                s = int(r["seed"])
                if s not in idx:
                    missing += 1
                    continue
                checked += 1
                for k in ("cubeA_x", "cubeA_y", "cubeA_theta",
                          "cubeB_x", "cubeB_y", "cubeB_theta"):
                    d = abs(float(r[k]) - float(idx[s][k]))
                    worst = max(worst, d)
                    if d > 1e-5:
                        mism += 1
                        break
            detail = f"{checked} episodes, max coord diff {worst:.2e}"
            if missing:
                detail += f", {missing} seeds not in the index"
            check(mism == 0 and checked == len(rows), f"mode '{tag}' block {b}",
                  detail)

    # ---- 5. one checkpoint produced all of it ------------------------------
    print("\n5. every block used the same checkpoint")
    shas = {}
    for f in sorted(glob.glob(f"{out}/mode_*_manifest.json")):
        m = json.load(open(f))
        if "ckpt_sha256" in m:
            shas.setdefault(m["ckpt_sha256"], []).append(os.path.basename(f))
    if not shas:
        check(False, "manifests carry ckpt_sha256",
              "no manifest found - re-run to get attribution")
    else:
        for h, files in shas.items():
            print(f"     {h}  {len(files)} block(s)")
        check(len(shas) == 1, "single checkpoint across all blocks",
              f"{len(shas)} distinct checkpoints" if len(shas) > 1 else "")

    # ---- 6. the deliverable's shape ----------------------------------------
    print("\n6. per-mode evaluation is 100 rollouts x 3 seeds")
    for tag in sorted(set(t for t, _, _, _ in found)):
        mine = [(b, rows) for t, b, _, rows in found if t == tag]
        ns = [len(rows) for _, rows in mine]
        pseeds = sorted(set(int(r["policy_seed"]) for _, rows in mine for r in rows))
        # One policy seed per block, and three distinct ones. Reusing a block
        # under three policy seeds would hold the initial states fixed and
        # measure DDPM sampling noise instead of a real error bar.
        per_block_one = all(len(set(int(r["policy_seed"]) for r in rows)) == 1
                            for _, rows in mine)
        check(len(mine) == 3 and all(n == 100 for n in ns)
              and len(pseeds) == 3 and per_block_one,
              f"mode '{tag}'",
              f"{len(mine)} blocks, sizes {ns}, policy seeds {pseeds}")

    # ---- 7. the numbers ----------------------------------------------------
    print("\n7. per-mode success rate  (mean +- SD over the 3 policy seeds)")
    print(f"   {'mode':>9} {'per-block rates':>22} {'mean':>7} {'SD':>7} "
          f"{'pooled 95% CI':>16} {'grasp':>7} {'place':>7} {'hold|pl':>8}  discovery")
    for tag in sorted(set(t for t, _, _, _ in found)):
        mine = [rows for t, _, _, rows in found if t == tag]
        rates = [sum(1 for r in b if r["success_once"] == "1") / len(b) for b in mine]
        rows = [r for b in mine for r in b]
        n = len(rows)
        k = sum(1 for r in rows if r["success_once"] == "1")
        g = sum(1 for r in rows if r["ever_grasped"] == "1")
        p = sum(1 for r in rows if r["ever_placed"] == "1")
        m = sum(rates) / len(rates)
        sd = (sum((x - m) ** 2 for x in rates) / (len(rates) - 1)) ** 0.5 \
            if len(rates) > 1 else float("nan")
        lo, hi = wilson(k, n)
        d = DISCOVERY.get(tag)
        note = ""
        if d:
            note = (f"{d[0]:.3f} [{d[1]:.3f}, {d[2]:.3f}] n={d[3]}"
                    + ("  ok" if d[1] <= m <= d[2] else "  <- outside"))
        print(f"   {tag:>9} {' '.join(f'{x:.3f}' for x in rates):>22} "
              f"{m:7.3f} {sd:7.3f} [{lo:.3f}, {hi:.3f}] "
              f"{g / n:7.3f} {p / n:7.3f} {k / max(p, 1):8.3f}  {note}")

    print("\n   'discovery' is what the 1,200-episode nominal pass predicted for"
          "\n   this region, pre-registered before these rollouts ran. Landing"
          "\n   outside it is informative, not a failure - it came from ~40"
          "\n   episodes and this is 300. What must HOLD is the mechanism split:"
          "\n   'gap' low on place with normal hold|place, 'farb' near-1.0 grasp"
          "\n   and normal place with hold|place collapsed.")

    print("")
    if FAILS:
        print(f"{len(FAILS)} CHECK(S) FAILED - do not report these numbers:")
        for f in FAILS:
            print(f"  - {f}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
