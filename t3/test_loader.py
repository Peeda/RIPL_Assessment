#!/usr/bin/env python
"""Layer A against every fixture. No deps, ~1 s.

    python3 t3/test_loader.py

THE PROBLEM THIS SOLVES. The thing under test is code an LLM wrote, so there is
no fixed input to write a test around - and "the checker returned no violations"
is not evidence that the checker checks anything. t2/test_verify.py solved the
same problem by committing twelve deliberate corruptions of a valid evaluation;
this commits ~30 deliberately-broken generations, each one a real mistake an LLM
makes, and asserts layer A rejects each FOR THE RIGHT REASON.

Matching the reason, not just the rejection, is the part that matters. A checker
with one over-broad rule rejects everything and passes a test that only counts
failures. Every row below names a substring the message must contain.

The fixtures that must PASS are as important: a gate nobody has seen accept
anything is a gate that might reject everything, and the two `good_*` files plus
the stock-reward transcription are the positive controls.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from loader import check_static, load_file  # noqa: E402

FIX = os.path.join(HERE, "fixtures")

# (fixture, kind, expected substring in the violation) - layer A must reject.
REJECT = [
    ("bad_syntax.py",            "reward",  "does not parse"),
    ("bad_import_os.py",         "reward",  "forbidden import 'os'"),
    ("bad_exec.py",              "reward",  "forbidden name 'eval'"),
    ("bad_open.py",              "reward",  "forbidden name 'open'"),
    ("bad_dunder.py",            "reward",  "dunder attribute"),
    ("bad_while_true.py",        "reward",  "`while` is not allowed"),
    ("bad_signature.py",         "reward",  "expected ('env', 'obs', 'action', 'info')"),
    ("bad_argorder.py",          "reward",  "names and order both matter"),
    ("bad_missing_max.py",       "reward",  "missing module-level 'REWARD_MAX"),
    ("bad_max_expression.py",    "reward",  "positive numeric literal"),
    ("bad_private_attr.py",      "reward",  "'env._episode_seed' is not on the allowed surface"),
    ("bad_elapsed_steps.py",     "reward",  "'env.elapsed_steps' is not on the allowed surface"),
    ("bad_item_sync.py",         "reward",  "'.item' is not allowed"),
    ("bad_mutates_env.py",       "reward",  "'.set_pose' is not allowed"),
    ("bad_class.py",             "reward",  "ClassDef"),
    ("bad_info_key.py",          "reward",  "is not a key evaluate() returns"),
    ("bad_sampler_numpy.py",     "sampler", "forbidden import 'numpy'"),
]

# Layer A must ACCEPT these. Each is broken, but broken in a way only a running
# simulator can see - which is the whole reason layers B through E exist. The
# comment is the layer that does catch it.
ACCEPT = [
    ("good_reward.py",             "reward"),   # -
    ("good_sampler.py",            "sampler"),  # -
    ("stock_reward.py",            "reward"),   # -
    ("bad_scalar.py",              "reward"),   # B: not a tensor
    ("bad_shape.py",               "reward"),   # B: (n, 1) not (n,)
    ("bad_nan.py",                 "reward"),   # B: divides by zero at contact
    ("bad_unbounded.py",           "reward"),   # B: leaves [0, REWARD_MAX]
    ("bad_nondeterministic.py",    "reward"),   # B: not pure
    ("hack_grasp_only.py",         "reward"),   # C + D: held-forever is optimal
    ("hack_proximity_xy.py",       "reward"),   # C: z sweep is flat
    ("hack_inverted.py",           "reward"),   # C: P4_inverted outscores success
    ("hack_success_only.py",       "reward"),   # C: no ordering has a margin
    ("bad_sampler_collapsed.py",   "sampler"),  # E: spread collapsed
    ("bad_sampler_offtable.py",    "sampler"),  # E: outside the nominal support
    ("bad_sampler_overlap.py",     "sampler"),  # E: cubes interpenetrate
    ("bad_sampler_unreachable.py", "sampler"),  # E: past the IK edge
    ("bad_sampler_tilted.py",      "sampler"),  # E: quaternion not about z
    ("bad_sampler_offregion.py",   "sampler"),  # E: hit rate at the base rate
]

FAILS = []


def check(ok, label, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILS.append(label)


def main():
    print("layer A rejects the broken generations\n")
    for name, kind, want in REJECT:
        with open(os.path.join(FIX, name)) as f:
            bad = check_static(f.read(), kind, name)
        if not bad:
            check(False, name, "layer A ACCEPTED it - the checker has a hole")
            continue
        hit = any(want in b for b in bad)
        check(hit, name,
              f"caught: {bad[0]}" if hit else
              f"rejected, but for the wrong reason. wanted {want!r}, got {bad}")

    print("\nlayer A accepts the well-formed ones (including the ones that are\n"
          "broken in ways only a simulator can see)\n")
    for name, kind in ACCEPT:
        with open(os.path.join(FIX, name)) as f:
            bad = check_static(f.read(), kind, name)
        check(not bad, name, "" if not bad else f"unexpectedly rejected: {bad}")

    print("\nthe positive controls import and expose a callable\n")
    for name, kind, fn in [("good_reward.py", "reward", "compute_reward"),
                           ("stock_reward.py", "reward", "compute_reward"),
                           ("good_sampler.py", "sampler", "sample_cube_poses")]:
        try:
            ns = load_file(os.path.join(FIX, name), kind)
            check(callable(ns[fn]), name, f"{fn} is callable")
        except Exception as e:                                   # noqa: BLE001
            # torch is not importable on the laptop, and that is fine: the
            # static half is what this file is testing. Anything else is real.
            if "No module named 'torch'" in str(e) or "torch" in str(e):
                print(f"  SKIP  {name}   no torch on this machine "
                      f"(static checks already passed)")
            else:
                check(False, name, f"{type(e).__name__}: {e}")

    print()
    if FAILS:
        print(f"{len(FAILS)} FAILED: {', '.join(FAILS)}")
        sys.exit(1)
    print(f"all {len(REJECT) + len(ACCEPT)} fixtures behaved as specified.")


if __name__ == "__main__":
    main()
