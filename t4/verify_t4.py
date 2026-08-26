#!/usr/bin/env python
"""Assert a T-IV before/after pair is what it claims. No sim, no numpy.

t2/verify.py already checks each pass on its own - reserved seeds, the schema,
disjointness, the recomputed geometry, the join against seeds.csv, one
checkpoint hash, the 3x100 shape. Run it on BOTH directories first; this adds
only what is invisible from inside one pass:

  1. the two passes are PAIRED - identical seed sets, block by block
  2. both ran on the SAME frozen base policy
  3. the after pass actually carried a residual, one per block
  4. residual seed b is in block b, which is what makes the spread include
     training variance rather than only DDPM sampling noise
  5. the residual differs between blocks - three training runs, not one file
     copied three times

Exits non-zero, so it can gate a report.

    python3 t4/verify_t4.py t2/results t4/results/after_gap
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "t2"))
from geometry import wilson  # noqa: E402

FAIL = []


def bad(msg):
    FAIL.append(msg)
    print(f"  FAIL  {msg}")


def load(d):
    """{(mode, block): (rows, manifest)} for one pass directory."""
    import csv
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "mode_*_seed[0-9].csv"))):
        stem = os.path.basename(p)[:-4]
        mode = stem[len("mode_"):stem.rindex("_seed")]
        block = int(stem[stem.rindex("_seed") + 5:])
        with open(p) as f:
            rows = list(csv.DictReader(f))
        mp = os.path.join(d, stem + "_manifest.json")
        man = json.load(open(mp)) if os.path.exists(mp) else {}
        out[(mode, block)] = (rows, man)
    return out


def rate(rows, key="success_once"):
    k = sum(int(r[key]) for r in rows)
    return k, len(rows)


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    before_dir, after_dir = sys.argv[1], sys.argv[2]
    before, after = load(before_dir), load(after_dir)
    if not before or not after:
        sys.exit(f"\n!! no mode_*_seed<n>.csv in "
                 f"{before_dir if not before else after_dir}\n")

    print(f"before  {before_dir}   {len(before)} blocks")
    print(f"after   {after_dir}   {len(after)} blocks\n")

    # -- 1. paired ---------------------------------------------------------
    print("1. the two passes are paired (identical seeds, block by block)")
    shared = sorted(set(before) & set(after))
    for k in sorted(set(after) - set(before)):
        bad(f"{k} is in the after pass but not the before pass")
    if not shared:
        bad("the two passes share no block at all")
    for key in shared:
        sb = [r["seed"] for r in before[key][0]]
        sa = [r["seed"] for r in after[key][0]]
        if sb != sa:
            extra, missing = set(sa) - set(sb), set(sb) - set(sa)
            bad(f"{key}: seed lists differ ({len(missing)} missing, "
                f"{len(extra)} extra). The comparison is two draws from the "
                f"region, not a paired before/after. Did seeds.csv get rebuilt "
                f"at a different INDEX_SEEDS?")
        elif len(sb) != len(set(sb)):
            bad(f"{key}: repeated seeds within a block")
    if not FAIL:
        print(f"   ok - {len(shared)} blocks, identical seed order")

    # -- 2. one frozen base ------------------------------------------------
    print("\n2. both passes ran the same frozen base policy")
    hashes = {m.get("ckpt_sha256") for _, m in list(before.values()) + list(after.values())}
    hashes.discard(None)
    if len(hashes) != 1:
        bad(f"{len(hashes)} distinct ckpt_sha256 across the two passes: "
            f"{sorted(hashes)}. A residual sitting on different base weights is "
            f"not a before/after of the same policy.")
    else:
        print(f"   ok - ckpt_sha256 {hashes.pop()} throughout")

    # -- 3/4/5. the residual ----------------------------------------------
    print("\n3. the after pass carried a residual, one per block")
    seen = {}
    for key in sorted(after):
        mode, block = key
        m = after[key][1]
        if not m.get("residual"):
            bad(f"{key}: the after manifest names no residual. This pass is the "
                f"BASE policy - RESIDUAL was unset or did not reach build_agent.")
            continue
        if not m.get("residual_sha256"):
            bad(f"{key}: residual recorded with no residual_sha256")
        seen[key] = m
    if seen and not FAIL:
        print(f"   ok - {len(seen)} blocks carry a residual")

    print("\n4/5. which design was run, and is it coherent")
    # Two legitimate designs, and they must not be silently mixed:
    #
    #   PAIRED   one residual per block, seed b in block b. The three block
    #            rates are then independent replicates of the whole pipeline
    #            and their SD carries TRAINING variance.
    #   SINGLE   one residual across all three blocks. The blocks then vary
    #            only the initial states and the DDPM noise, so the SD does
    #            NOT carry training variance and the report must say so.
    #
    # Anything in between - two distinct residuals across three blocks - is
    # neither, and is far more likely to be a mis-set $RESIDUAL than a choice.
    by_mode = {}
    for (mode, block), m in seen.items():
        by_mode.setdefault(mode, {})[block] = (m.get("residual_sha256"),
                                               m.get("residual_seed"))
    designs = {}
    for mode, d in sorted(by_mode.items()):
        shas = {v[0] for v in d.values()}
        n = len(d)
        if len(shas) == n:
            designs[mode] = "paired"
            for block, (_, rs) in sorted(d.items()):
                if rs is None:
                    bad(f"({mode}, {block}): no residual_seed, so the pairing "
                        f"cannot be checked")
                elif int(rs) != block:
                    bad(f"({mode}, {block}): block {block} ran residual seed "
                        f"{rs}. Pairing seed b with block b is the whole point "
                        f"of the paired design.")
            print(f"   ok - '{mode}': PAIRED, {n} residuals over {n} blocks")
        elif len(shas) == 1:
            designs[mode] = "single"
            seeds = {v[1] for v in d.values()}
            if len(seeds) != 1:
                bad(f"mode '{mode}': one residual hash but {len(seeds)} "
                    f"residual_seed values - the manifests disagree with the "
                    f"weights.")
            print(f"   ok - '{mode}': SINGLE residual (seed {seeds.pop()}) "
                  f"over {n} blocks")
        else:
            bad(f"mode '{mode}': {len(shas)} distinct residuals across {n} "
                f"blocks - neither the paired design (one per block) nor the "
                f"single-residual design. That is almost certainly a mis-set "
                f"$RESIDUAL rather than a choice.")
    if any(v == "single" for v in designs.values()):
        print("\n   NOTE: a SINGLE residual was evaluated across all three "
              "blocks, so the\n   block-to-block SD reported below reflects "
              "initial states and DDPM\n   sampling only. TRAINING variance is "
              "NOT in it. Say so in the report.")

    # -- the table ---------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"{'mode':>9}  {'before':>26}   {'after':>26}   {'delta':>7}")
    modes = sorted({m for m, _ in shared})
    for mode in modes:
        blocks = sorted(b for m, b in shared if m == mode)
        cells = []
        for src in (before, after):
            per = [rate(src[(mode, b)][0])[0] / rate(src[(mode, b)][0])[1]
                   for b in blocks]
            k = sum(rate(src[(mode, b)][0])[0] for b in blocks)
            n = sum(rate(src[(mode, b)][0])[1] for b in blocks)
            lo, hi = wilson(k, n)
            cells.append((per, k / n, lo, hi))
        (pb, mb, lb, hb), (pa, ma, la, ha) = cells
        print(f"{mode:>9}  {' '.join(f'{x:.3f}' for x in pb)} "
              f"{mb:.3f} [{lb:.3f},{hb:.3f}]   "
              f"{' '.join(f'{x:.3f}' for x in pa)} "
              f"{ma:.3f} [{la:.3f},{ha:.3f}]   {ma - mb:+.3f}")
    print("\n  Overlapping Wilson intervals are NOT proof of no effect, and "
          "disjoint\n  ones are not proof of one - at n=300 pooled the interval "
          "is ~+-0.05, so\n  'near-zero degradation' is a claim this shape "
          "supports only weakly. Say so.")

    print("")
    if FAIL:
        print(f"{len(FAIL)} check(s) FAILED")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
