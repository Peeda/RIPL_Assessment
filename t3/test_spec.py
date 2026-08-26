#!/usr/bin/env python
"""The contract is self-consistent and the statistics are right. No deps, ~1 s.

    python3 t3/test_spec.py

t3/spec.py is read by three consumers that never meet: the prompt the model
sees, the checker that lints what it writes, and the gate that decides whether
the result is usable. This file is what stops them drifting - it renders the
prompt text and asserts that everything the checker enforces actually appears in
it, so a rule cannot be tightened without the model being told.

It also checks the arithmetic behind the thresholds. An AUC threshold is only
meaningful next to the sample size it will be applied at, and "AUC >= 0.75"
sounds equally reasonable at n = 100 and at n = 30 while meaning something
completely different. The power table below is computed, not asserted from
memory, so the numbers in t3/README.md are checked every time this runs.
"""
import csv
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(os.path.dirname(HERE), "t2"))
import spec  # noqa: E402

FAILS = []


def check(ok, label, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILS.append(label)


def close(a, b, tol=1e-9):
    return abs(a - b) < tol


def test_prompt_renders():
    contract = spec.contract_markdown()
    api = spec.api_surface_markdown()

    for text, name in ((contract, "contract_markdown"), (api, "api_surface_markdown")):
        check("{{" not in text, f"{name} leaves no placeholder")

    # Everything the checker enforces has to be stated in the prompt.
    for fn, params in list(spec.REWARD_MODULE_REQUIRES.items()) + \
                      list(spec.SAMPLER_MODULE_REQUIRES.items()):
        check(fn in contract, f"contract names {fn}()")
        check(all(p in contract for p in params),
              f"contract names {fn}'s parameters", ", ".join(params))
    check(spec.REWARD_MAX_NAME in contract, "contract names REWARD_MAX")
    missing = [a for a in spec.ALLOWED_ENV_ATTRS if a not in api]
    check(not missing, "api surface lists every allowed env attribute",
          f"missing {missing}" if missing else f"{len(spec.ALLOWED_ENV_ATTRS)} entries")
    missing = [k for k in spec.ALLOWED_INFO_KEYS if k in api]
    check(len(missing) == len(spec.ALLOWED_INFO_KEYS),
          "api surface lists every allowed info key")
    check(f"{spec.MIN_SEPARATION:.5f}" in contract,
          "contract states the separation floor", f"{spec.MIN_SEPARATION:.5f} m")

    # The prompt files on disk must have no placeholder the assembler cannot
    # fill. Catches a renamed splice point, which would otherwise reach the
    # model as a literal {{CONTRACT}}.
    import re
    known = {"{{CONTRACT}}", "{{API_SURFACE}}", "{{ENV_SOURCE}}",
             "{{N_FRAMES}}", "{{SEED}}", "{{OUTCOME}}"}
    for name in sorted(os.listdir(os.path.join(HERE, "prompts"))):
        with open(os.path.join(HERE, "prompts", name)) as f:
            found = set(re.findall(r"\{\{[A-Z_]+\}\}", f.read()))
        check(found <= known, f"prompts/{name} uses only known placeholders",
              f"unknown: {found - known}" if found - known else
              (", ".join(sorted(found)) or "none"))


def test_loader_shares_the_spec():
    import loader
    check(loader.ALLOWED_ENV_ATTRS is spec.ALLOWED_ENV_ATTRS,
          "loader uses spec's attribute surface (identity, not a copy)")
    check(loader.ALLOWED_IMPORTS is spec.ALLOWED_IMPORTS,
          "loader uses spec's import whitelist")


def test_auc():
    # A perfect separator, a constant, and a reversal.
    check(close(spec.auc([1, 2, 3, 4], [0, 0, 1, 1]), 1.0), "auc: perfect = 1.0")
    check(close(spec.auc([4, 3, 2, 1], [0, 0, 1, 1]), 0.0), "auc: reversed = 0.0")
    check(close(spec.auc([1, 1, 1, 1], [0, 0, 1, 1]), 0.5), "auc: all ties = 0.5")
    # One tie across the classes: 1 of 4 pairs ties, 2 clean wins, 1 clean loss.
    #   pos = [2, 3], neg = [1, 2] -> (2>1) + (2~2)/2 + (3>1) + (3>2) = 3.5/4
    check(close(spec.auc([2, 3, 1, 2], [1, 1, 0, 0]), 0.875),
          "auc: half credit for a tie", "0.875 by hand")
    check(spec.auc([1, 2], [1, 1]) != spec.auc([1, 2], [1, 1]),
          "auc: nan when a class is empty", "nan != nan")

    # AUC is U / (n1 * n0). Cross-check against the brute-force pair count.
    import random
    rnd = random.Random(0)
    for _ in range(200):
        n = rnd.randint(4, 20)
        s = [rnd.randint(0, 5) for _ in range(n)]
        y = [rnd.randint(0, 1) for _ in range(n)]
        if not (0 < sum(y) < n):
            continue
        wins = sum((a > b) + 0.5 * (a == b)
                   for a, ya in zip(s, y) if ya
                   for b, yb in zip(s, y) if not yb)
        want = wins / (sum(y) * (n - sum(y)))
        if not close(spec.auc(s, y), want, 1e-9):
            check(False, "auc == U/(n1*n0) on random cases",
                  f"{spec.auc(s, y)} vs {want}")
            return
    check(True, "auc == U/(n1*n0) on 200 random cases (ties included)")


def test_conditional_and_pb():
    scores = [1, 5, 2, 6, 9, 9]
    labels = [0, 1, 0, 1, 1, 0]
    given = [1, 1, 1, 1, 0, 0]
    a, n1, n0 = spec.conditional_auc(scores, labels, given)
    check(close(a, 1.0) and n1 == 2 and n0 == 2,
          "conditional_auc restricts to the stratum", f"auc {a} n={n1}/{n0}")
    a, n1, n0 = spec.conditional_auc(scores, labels, [0] * 6)
    check(a != a and n1 == 0, "conditional_auc: nan on an empty stratum")
    check(close(spec.point_biserial([1, 2, 3, 4], [0, 0, 1, 1]), 0.8944, 1e-4),
          "point_biserial matches Pearson r against a 0/1 vector")


def test_power():
    """The declared thresholds have to be detectable at the declared n.

    Recomputed here rather than trusted: this is the table t3/README.md quotes,
    and a threshold that reads as strict while being unfalsifiable at the sample
    size it runs at is worse than no threshold.
    """
    print("\n  power of the layer-D thresholds, at T-II's measured stage rates")
    print(f"    {'test':<28} {'n1/n0':>9} {'SE_null':>8} {'z at min':>9}")
    rows = [
        ("unconditional, n=100", 52, 48, spec.ALIGN_AUC_MIN),
        ("unconditional, n=60", 31, 29, spec.ALIGN_AUC_MIN),
        ("unconditional, n=30", 16, 14, spec.ALIGN_AUC_MIN),
        ("gap  placed|grasped n=100", 63, 33, spec.ALIGN_COND_AUC_MIN),
        ("farb success|placed n=100", 56, 29, spec.ALIGN_COND_AUC_MIN),
    ]
    z = {}
    for lab, n1, n0, thr in rows:
        se = spec.se_null_auc(n1, n0)
        zz = (thr - 0.5) / se
        z[lab] = zz
        print(f"    {lab:<28} {f'{n1}/{n0}':>9} {se:>8.3f} {zz:>9.1f}")
    check(z["unconditional, n=100"] >= spec.ALIGN_Z_MIN,
          f"at n={spec.ALIGN_N_TARGET} the AUC threshold clears z>={spec.ALIGN_Z_MIN}")
    check(z["unconditional, n=60"] >= spec.ALIGN_Z_MIN,
          f"at the floor n={spec.ALIGN_N_MIN} it still clears it")
    check(z["unconditional, n=30"] < spec.ALIGN_Z_MIN,
          "at n=30 it does NOT - which is why the gate refuses fewer than "
          f"{spec.ALIGN_N_MIN}")
    for lab in ("gap  placed|grasped n=100", "farb success|placed n=100"):
        check(z[lab] >= 2.9, f"the conditional threshold has power: {lab}",
              f"z {z[lab]:.1f}")


def test_base_rates():
    """The nominal region rates, recomputed from the committed seed index.

    The same discipline t2/test_geometry.py applies to DISCOVERY: a number the
    report quotes is checked against the evidence it came from, so it cannot go
    stale.
    """
    from geometry import MODES, geom_from_row
    path = os.path.join(os.path.dirname(HERE), "t2", "results", "seeds.csv")
    if not os.path.exists(path):
        print(f"  SKIP  base rates - {path} not present")
        return
    with open(path) as f:
        rows = list(csv.DictReader(f))
    print(f"\n  nominal region rates over {len(rows)} indexed seeds")
    for tag in ("gap", "farb"):
        hits = sum(1 for r in rows if MODES[tag](geom_from_row(r)))
        rate = hits / len(rows)
        print(f"    {tag:<6} {hits:>5}/{len(rows)} = {rate:.4f}   "
              f"a sampler at the {spec.SAMPLER_HIT_RATE_MIN} floor is "
              f"{spec.SAMPLER_HIT_RATE_MIN / rate:.0f}x this")
        check(spec.SAMPLER_HIT_RATE_MIN / rate >= spec.SAMPLER_ENRICHMENT_MIN,
              f"the hit-rate floor implies the enrichment floor for '{tag}'",
              f"{spec.SAMPLER_HIT_RATE_MIN / rate:.0f}x >= "
              f"{spec.SAMPLER_ENRICHMENT_MIN}x")


def test_constants():
    check(close(spec.MIN_SEPARATION, 2 * (math.sqrt(0.0008) + 0.001), 1e-12),
          "MIN_SEPARATION is the environment's own floor",
          f"{spec.MIN_SEPARATION:.5f} m for 40 mm cubes")
    check(spec.REACH_MAX > 0.76,
          "REACH_MAX sits above mode farb's dist_B threshold",
          f"{spec.REACH_MAX} > 0.76 - a lower cap would exclude the region "
          f"it is meant to guard")
    ids = [p for p, _ in spec.PROBES]
    check(len(ids) == len(set(ids)), "probe ids are unique")
    used = {p for o in spec.PROBE_ORDERINGS for p in o[:2]}
    check(used <= set(ids), "every ordering names a probe that exists",
          f"unknown: {used - set(ids)}" if used - set(ids) else f"{len(used)} used")
    check("P0_success" in {o[0] for o in spec.PROBE_ORDERINGS},
          "the completed stack is the reference in the orderings")
    for m, s in spec.MODE_STAGE.items():
        check(s["given"] in spec.ALIGN_COLUMNS and s["target"] in spec.ALIGN_COLUMNS,
              f"mode '{m}' conditions on columns that are logged",
              f"{s['target']} | {s['given']}")


def main():
    for t in (test_prompt_renders, test_loader_shares_the_spec, test_auc,
              test_conditional_and_pb, test_constants, test_power,
              test_base_rates):
        t()
    print()
    if FAILS:
        print(f"{len(FAILS)} FAILED: {', '.join(FAILS)}")
        sys.exit(1)
    print("the contract, the prompt rendering and the statistics agree.")


if __name__ == "__main__":
    main()
