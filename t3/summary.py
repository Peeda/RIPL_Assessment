#!/usr/bin/env python
"""Read what t3/check.py measured, print it, and get out of the way. Stdlib.

    python3 t3/summary.py $T3_OUT --mode gap [--run t3/artifacts/gap]

THIS IS NOT A GATE, AND THAT IS THE DESIGN.
    An earlier version of this file exited non-zero when any threshold was
    missed. That is the wrong instrument here: an LLM-written reward that scores
    0.72 where the threshold says 0.75 is a thing to write a sentence about, not
    a thing to block T-IV on, and a refusal costs a regeneration cycle to learn
    something the number already told you.

    So every threshold prints WARN and this exits 0. The one exception is an
    artifact that could not be loaded at all, which is not a judgement call.

    A WARN is a finding. Put it in the report; that is what it is for.

Pure stdlib - no numpy, no torch, no ManiSkill - so the numbers that get quoted
in the write-up can be re-derived on a laptop from the committed CSVs. It is a
separate file from check.py for exactly that reason: check.py must import
env_t3 at module scope for the forkserver workers, which drags ManiSkill in.
"""
import argparse
import csv
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from spec import (ALIGN_AUC_MIN, ALIGN_N_TARGET, ALIGN_STAGE_GAP_FRAC,  # noqa: E402
                  ALIGN_Z_MIN, REWARD_FILE, SAMPLER_ENRICHMENT_MIN,
                  SAMPLER_FILE, SAMPLER_HIT_RATE_MIN, SAMPLER_SD_MIN_XY,
                  auc, auc_z, mean)

N_OK = N_WARN = 0
BLOCKING = []


def ok(label, detail=""):
    global N_OK
    N_OK += 1
    print(f"    OK    {label}" + (f"   {detail}" if detail else ""))


def warn(label, detail="", extra=""):
    global N_WARN
    N_WARN += 1
    print(f"    WARN  {label}" + (f"   {detail}" if detail else ""))
    if extra:
        print(f"          {extra}")


def rate(cond, label, detail="", extra=""):
    ok(label, detail) if cond else warn(label, detail, extra)


def note(label, detail=""):
    print(f"    ....  {label}" + (f"   {detail}" if detail else ""))


def _rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _f(r, k):
    v = r.get(k, "")
    return float(v) if v not in ("", None) else float("nan")


# ---------------------------------------------------------------------------


def section_reward(out, label):
    path = os.path.join(out, f"reward_{label}.json")
    print("\n  reward.py")
    if not os.path.exists(path):
        note("not measured", "`bash t3/run.sh check`")
        return None
    m = json.load(open(path))
    rate(m["shape_ok"] and m["dtype_ok"] and m["finite_ok"] and m["device_ok"],
         "shape (num_envs,), floating, finite, right device",
         f"{m['n_calls']} calls", m.get("note", ""))
    rate(m["pure_ok"], "pure - same state, same value", "",
         "a stray torch.rand in a shaping term makes every advantage estimate "
         "noisy")
    rate(m["mutation_ok"], "does not mutate the simulator", "",
         "the reward changed env.get_state() - it is moving the thing it scores")
    rate(m["bounds_ok"], "stays inside [0, REWARD_MAX]",
         f"range [{m['r_min']:.3f}, {m['r_max']:.3f}] of {m['reward_max']}",
         "the env divides by REWARD_MAX, so PPO sees a normalised reward "
         "outside [0, 1]")
    return m


def section_sampler(out, mode):
    path = os.path.join(out, f"sampler_{mode}.json")
    print("\n  sampler.py")
    if not os.path.exists(path):
        note("not measured", "`bash t3/run.sh check`")
        return None
    m = json.load(open(path))
    bad = {k: m[k] for k in ("quat_bad", "z_bad", "separation_bad",
                             "support_bad", "reach_bad") if m[k]}
    rate(not bad, "every draw is a state the environment could have produced",
         f"{m['draws']} draws" if not bad else f"{bad}",
         "" if not bad else "draws outside the nominal support make T-IV a "
                            "distribution-shift problem, not a failure-mode one")
    rate(m["deterministic"], "deterministic under a fixed torch seed", "",
         "it is drawing from something other than torch - reset(seed=s) is no "
         "longer reproducible, so T-IV's training distribution cannot be "
         "regenerated")
    if m.get("enrichment"):
        rate(m["hit_rate"] >= SAMPLER_HIT_RATE_MIN
             and m["enrichment"] >= SAMPLER_ENRICHMENT_MIN,
             "biased toward the failure region",
             f"hit rate {m['hit_rate']:.3f} vs nominal {m['base_rate']:.4f} "
             f"-> {m['enrichment']:.1f}x  "
             f"(want {SAMPLER_HIT_RATE_MIN}, {SAMPLER_ENRICHMENT_MIN}x)")
    else:
        rate(m["hit_rate"] >= SAMPLER_HIT_RATE_MIN,
             "draws land in the failure region",
             f"{m['hit_rate']:.3f} (want {SAMPLER_HIT_RATE_MIN}); no seed "
             f"index, so enrichment is unknown")
    tight = [k for k in ("sd_cubeA_x", "sd_cubeA_y", "sd_cubeB_x", "sd_cubeB_y")
             if m[k] < SAMPLER_SD_MIN_XY]
    rate(not tight, "the draws are varied",
         f"sd(A) {m['sd_cubeA_x']:.3f}/{m['sd_cubeA_y']:.3f}  "
         f"sd(B) {m['sd_cubeB_x']:.3f}/{m['sd_cubeB_y']:.3f} m",
         "" if not tight else f"{tight} below {SAMPLER_SD_MIN_XY} m - a sampler "
                              f"concentrated on one configuration hits the "
                              f"region perfectly and teaches T-IV one state")
    return m


def align_stats(rows):
    pol = [r for r in rows if r["arm"] == "policy"]
    if not pol:
        return None
    ret = [_f(r, "ep_return") for r in pol]
    suc = [r["success_once"] == "1" for r in pol]
    n1 = sum(suc)
    a = auc(ret, suc)
    groups = {
        "grasped, not placed": [x for x, r in zip(ret, pol)
                                if r["ever_grasped"] == "1"
                                and r["ever_placed"] != "1"],
        "placed, not success": [x for x, r in zip(ret, pol)
                                if r["ever_placed"] == "1"
                                and r["success_once"] != "1"],
        "success": [x for x, r in zip(ret, pol) if r["success_once"] == "1"],
    }
    return dict(n=len(pol), n1=n1, n0=len(pol) - n1, auc=a,
                z=auc_z(a, n1, len(pol) - n1),
                rng=(max(ret) - min(ret)) if ret else float("nan"),
                groups={k: mean(v) for k, v in groups.items()},
                group_n={k: len(v) for k, v in groups.items()},
                arms={arm: mean(_f(r, "ep_return")
                                for r in rows if r["arm"] == arm)
                      for arm in sorted({r["arm"] for r in rows})})


def section_align(out, label):
    path = os.path.join(out, f"align_{label}.csv")
    print("\n  alignment")
    if not os.path.exists(path):
        note("not measured", "`bash t3/run.sh check`")
        return None
    s = align_stats(_rows(path))
    if s is None:
        note("no policy-arm episodes in the CSV")
        return None

    rate(s["n"] >= ALIGN_N_TARGET, "enough episodes",
         f"{s['n']} ({s['n1']} success / {s['n0']} fail)")
    rate(s["auc"] >= ALIGN_AUC_MIN and s["z"] >= ALIGN_Z_MIN,
         "ranks successes above failures",
         f"AUC {s['auc']:.3f}  z {s['z']:.1f}  "
         f"(want {ALIGN_AUC_MIN}, z {ALIGN_Z_MIN})",
         "" if s["auc"] >= ALIGN_AUC_MIN else
         "the reward cannot separate observed success from observed failure on "
         "the policy we already have, so it will not shape a residual toward it")

    # THE stage test. See spec.ALIGN_STAGE_GAP_FRAC: a stage-conditional AUC has
    # a structural floor and cannot catch a grasp-farming reward; requiring the
    # mean return to RISE from stage to stage can.
    g, gn = s["groups"], s["group_n"]
    order = ["grasped, not placed", "placed, not success", "success"]
    have = [k for k in order if gn[k] > 0]
    vals = [g[k] for k in have]
    need = ALIGN_STAGE_GAP_FRAC * s["rng"]
    steps = list(zip(vals, vals[1:]))
    bad = [i for i, (x, y) in enumerate(steps) if y < x + need]
    rate(not bad, "mean return rises with the stage the episode reached",
         "  <  ".join(f"{k}={g[k]:.1f}(n={gn[k]})" for k in have)
         + f"   [each step needs +{need:.1f}]",
         "" if not bad else
         f"the step {have[bad[0]]} -> {have[bad[0] + 1]} does not rise: the "
         f"reward pays more for the earlier stage, which is what a "
         f"grasp-farming reward looks like")

    for arm in ("jitter", "zero"):
        if arm in s["arms"]:
            rate(s["arms"]["policy"] > s["arms"][arm],
                 f"the real policy out-earns `{arm}`",
                 f"{s['arms']['policy']:.1f} vs {s['arms'][arm]:.1f}")
    return s


def section_manifest(out, label, run):
    path = os.path.join(out, f"align_{label}_manifest.json")
    if not os.path.exists(path):
        return
    m = json.load(open(path))
    # Not a threshold - a measurement over the wrong population. With the biased
    # sampler on, reset(seed=s) does not reproduce the T-II episode for s, so
    # every number above would be a plausible answer to a different question.
    if int(m.get("t3_sampler", -1)) != 0:
        warn("the biased sampler was ON during the alignment measurement",
             f"t3_sampler={m.get('t3_sampler')}",
             "these are NOT the T-II seeds' states - rerun with T3_SAMPLER=0")
    ck = {json.load(open(p)).get("ckpt_sha256")
          for p in glob.glob(os.path.join(out, "align_*_manifest.json"))}
    if len(ck) > 1:
        warn("more than one checkpoint behind the arms", f"{len(ck)} sha256s",
             "the generated and stock arms are not comparable")


def calibration(out, gen):
    path = os.path.join(out, "align_stock.csv")
    if gen is None:
        return
    if not os.path.exists(path):
        note("stock-reward arm not run", "`bash t3/run.sh calibrate` - so the "
                                         "AUC above is uncalibrated")
        return
    s = align_stats(_rows(path))
    if s is None:
        return
    print(f"\n  calibration against ManiSkill's own 8-stage dense reward, "
          f"same {s['n']} episodes")
    print(f"    {'':<18} {'generated':>10} {'stock':>10}")
    print(f"    {'AUC (success)':<18} {gen['auc']:>10.3f} {s['auc']:>10.3f}")
    print(f"    {'z':<18} {gen['z']:>10.1f} {s['z']:>10.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--mode", required=True)
    ap.add_argument("--label")
    ap.add_argument("--run")
    a = ap.parse_args()
    label = a.label or a.mode
    print(f"t3 summary   mode '{a.mode}'   out {a.out}"
          + (f"   run {a.run}" if a.run else ""))

    if a.run:
        from loader import check_static
        print("\n  artifacts")
        for fname, kind in ((REWARD_FILE, "reward"), (SAMPLER_FILE, "sampler")):
            p = os.path.join(a.run, fname)
            if not os.path.exists(p):
                BLOCKING.append(f"{fname} is missing from {a.run}")
                print(f"    ----  {fname}   missing")
                continue
            with open(p) as f:
                errors, warnings = check_static(f.read(), kind, fname)
            if errors:
                BLOCKING.append(f"{fname}: {errors[0]}")
                print(f"    ----  {fname}   {len(errors)} contract error(s)")
                for e in errors:
                    print(f"          {e}")
            else:
                ok(f"{fname} satisfies the contract")
            for w in warnings:
                warn(f"{fname}", w)

    section_reward(a.out, label)
    section_sampler(a.out, a.mode)
    gen = section_align(a.out, label)
    section_manifest(a.out, label, a.run)
    calibration(a.out, gen)

    if gen is not None:
        with open(os.path.join(a.out, f"summary_{a.mode}.json"), "w") as f:
            json.dump(dict(mode=a.mode, label=label, n_ok=N_OK, n_warn=N_WARN,
                           blocking=BLOCKING,
                           auc=gen["auc"], z=gen["z"], n=gen["n"],
                           stage_means=gen["groups"], stage_n=gen["group_n"]),
                      f, indent=2)

    print(f"\n  {N_OK} OK, {N_WARN} WARN, {len(BLOCKING)} blocking.")
    if BLOCKING:
        for b in BLOCKING:
            print(f"    {b}")
        print("\n  An artifact that will not load is the one thing this refuses. "
              "Keep the\n  generation - which check caught it is the report's "
              "account of how an\n  LLM-written reward fails. Tune "
              "t3/prompts/hacking.md and regenerate.")
        sys.exit(1)
    if N_WARN:
        print("  WARNs are findings for the report, not a refusal. Read them, "
              "write them\n  up, and decide whether to regenerate or to train "
              "on this and say why.")


if __name__ == "__main__":
    main()
