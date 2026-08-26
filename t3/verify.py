#!/usr/bin/env python
"""The T-III gate. Reads files, decides, exits non-zero. Pure stdlib.

    python3 t3/verify.py $T3_OUT --mode gap [--run t3/artifacts/gap]

Nothing in T-III is reportable until this exits 0.

WHY THE DECISION IS SEPARATE FROM THE MEASUREMENT
    probes.py, sampler_check.py and align.py all need a simulator, a checkpoint
    and a GPU. If the thresholds lived inside them, there would be no way to
    test that the gate catches anything without spending a pod session per test
    case - and a gate nobody has tested is a gate that reports PASS.

    So every sim-touching script WRITES NUMBERS TO A FILE and this one reads
    them. t3/test_verify.py can then fabricate a complete, valid-looking T-III
    pass, corrupt it sixteen ways - a reward that scores a held cube above a
    completed stack, an alignment CSV whose returns have been shuffled against
    the outcomes, a sampler collapsed onto one pose - and confirm each is
    caught, on a laptop, in two seconds. That is the same structure t2/verify.py
    and t2/test_verify.py have, for the same reason.

WHAT EACH CHECK CORRESPONDS TO
    Every one of them is a way an LLM-generated reward has been, or could
    quietly be, wrong:

      * a reward whose maximum is reached while the cube is still held, when the
        task's own success criterion requires letting go;
      * one that pays for reaching and grasping and separates outcomes only
        because grasping predicts everything downstream;
      * one with the goal arguments swapped, which no rollout can reveal because
        the policy never produces the state that would expose it;
      * a sampler that hits the target region perfectly by returning one
        configuration;
      * a measurement taken with the biased sampler accidentally left on, which
        yields a plausible number over the wrong population.
"""
import argparse
import csv
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(os.path.dirname(HERE), "t2"))
from spec import (ALIGN_AUC_MIN, ALIGN_COND_AUC_MIN, ALIGN_COND_N_MIN,  # noqa: E402
                  ALIGN_COLUMNS, ALIGN_CONTROL_RATIO, ALIGN_MEAN_GAP_FRAC,
                  ALIGN_N_MIN, ALIGN_STAGE_GAP_FRAC, ALIGN_Z_MIN,
                  MODE_STAGE, PROBE_ORDERINGS,
                  REWARD_FILE, SAMPLER_COVER_FRAC, SAMPLER_DISTINCT_FRAC,
                  SAMPLER_ENRICHMENT_MIN, SAMPLER_FILE, SAMPLER_HIT_RATE_MIN,
                  SAMPLER_SD_MIN_GAP, SAMPLER_SD_MIN_XY, SAMPLER_YAW_BINS,
                  SAMPLER_YAW_BINS_MIN, SWEEP_MAX_VIOLATIONS, SWEEP_RANGE_MIN,
                  auc, auc_z, conditional_auc, mean, point_biserial)

FAILS = []


def check(ok, label, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILS.append(label)
    return ok


def note(label, detail):
    print(f"  ....  {label}   {detail}")


def _rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _f(r, k):
    v = r.get(k, "")
    return float(v) if v not in ("", None) else float("nan")


# ---------------------------------------------------------------------------
# 1 - layer A, re-run on the artifacts themselves
# ---------------------------------------------------------------------------


def check_static_artifacts(run):
    if not run or not os.path.isdir(run):
        note("1 static", f"no --run given, skipping the re-lint")
        return
    from loader import check_static
    for fname, kind in ((REWARD_FILE, "reward"), (SAMPLER_FILE, "sampler")):
        path = os.path.join(run, fname)
        if not os.path.exists(path):
            check(False, f"1 static  {fname}", "missing")
            continue
        with open(path) as f:
            bad = check_static(f.read(), kind, fname)
        check(not bad, f"1 static  {fname}",
              "contract satisfied" if not bad else f"{len(bad)}: {bad[0]}")


# ---------------------------------------------------------------------------
# 2 - layer B
# ---------------------------------------------------------------------------


def check_layer_b(out, label):
    path = os.path.join(out, f"probes_{label}.json")
    if not os.path.exists(path):
        check(False, "2 runtime", f"{path} missing - run `bash t3/run.sh probes`")
        return None
    j = json.load(open(path))
    b = j["layer_b"]
    for key, lab in (("shape_ok", "returns (num_envs,)"),
                     ("dtype_ok", "floating dtype"),
                     ("finite_ok", "finite everywhere"),
                     ("device_ok", "on the env's device"),
                     ("bounds_ok", f"within [0, {b['reward_max']}]"),
                     ("pure_ok", "pure - same state, same value"),
                     ("mutation_ok", "does not mutate the simulator")):
        check(bool(b.get(key)), f"2 runtime  {lab}",
              b.get("note", "") if not b.get(key) else
              (f"range [{b['r_min']:.3f}, {b['r_max']:.3f}] over "
               f"{b['n_calls']} calls" if key == "bounds_ok" else ""))
    check(not b.get("bad_attrs"), "2 runtime  stays on the allowed surface",
          f"read {b['bad_attrs']}" if b.get("bad_attrs") else
          f"touched {', '.join(b['touched'])}")
    return j


# ---------------------------------------------------------------------------
# 3 - layer C
# ---------------------------------------------------------------------------


def check_layer_c(out, label, reward_max, notes):
    path = os.path.join(out, f"probes_{label}.csv")
    if not os.path.exists(path):
        check(False, "3 probes", f"{path} missing")
        return
    r = {row["probe"]: float(row["reward"]) for row in _rows(path)}
    flags = {row["probe"]: row for row in _rows(path)}

    # The reference must actually BE a success, or every ordering below is
    # comparing against a mislabelled state.
    if "P0_success" in flags:
        check(flags["P0_success"]["success"] == "1",
              "3 probes  P0_success really is a success state",
              "evaluate() agrees" if flags["P0_success"]["success"] == "1"
              else "evaluate() says it is NOT - the probe is mislabelled and "
                   "every ordering below is meaningless")
    if "P7_held" in flags:
        check(flags["P7_held"]["is_cubeA_grasped"] == "1",
              "3 probes  P7_held really is grasped",
              "contact forces confirm the grasp")

    for hi, lo, margin in PROBE_ORDERINGS:
        if hi not in r or lo not in r:
            missing = hi if hi not in r else lo
            check(False, f"3 probes  {hi} > {lo}",
                  f"{missing} was not measured: "
                  f"{notes.get(missing, 'absent from the CSV')}")
            continue
        need = margin * reward_max
        ok = r[hi] >= r[lo] + need
        check(ok, f"3 probes  {hi} > {lo} + {margin:.0%}",
              f"{r[hi]:.3f} vs {r[lo]:.3f} (needs +{need:.3f})")

    spath = os.path.join(out, f"sweeps_{label}.csv")
    if not os.path.exists(spath):
        check(False, "3 sweeps", f"{spath} missing")
        return
    rows = _rows(spath)
    for name in sorted({x["sweep"] for x in rows}):
        s = sorted((x for x in rows if x["sweep"] == name),
                   key=lambda x: int(x["i"]))
        vals = [float(x["reward"]) for x in s]
        viol = sum(1 for a, b in zip(vals, vals[1:]) if b > a + 1e-9)
        check(viol <= SWEEP_MAX_VIOLATIONS,
              f"3 sweeps  {name}: reward does not rise as the cube moves away",
              f"{viol} increases (allowed {SWEEP_MAX_VIOLATIONS})")
        rng = (max(vals) - min(vals)) / reward_max
        # A reward that ignores this axis is perfectly monotone along it because
        # it is CONSTANT along it. Requiring a real range is what catches an
        # xy-only distance term.
        check(rng >= SWEEP_RANGE_MIN,
              f"3 sweeps  {name}: the reward actually varies along this axis",
              f"range {rng:.3f} of max (needs {SWEEP_RANGE_MIN})")


# ---------------------------------------------------------------------------
# 4 - layer D
# ---------------------------------------------------------------------------


def align_stats(rows, mode):
    pol = [r for r in rows if r["arm"] == "policy"]
    ret = [_f(r, "ep_return") for r in pol]
    suc = [r["success_once"] == "1" for r in pol]
    n1, n0 = sum(suc), len(suc) - sum(suc)

    a = auc(ret, suc)
    stage = MODE_STAGE.get(mode, MODE_STAGE["nominal"])
    given = [r[stage["given"]] == "1" for r in pol]
    target = [r[stage["target"]] == "1" for r in pol]
    ca, cn1, cn0 = conditional_auc(ret, target, given)

    rng = (max(ret) - min(ret)) if ret else float("nan")
    m1 = mean(x for x, y in zip(ret, suc) if y)
    m0 = mean(x for x, y in zip(ret, suc) if not y)

    groups = {
        "grasped, not placed": [x for x, r in zip(ret, pol)
                                if r["ever_grasped"] == "1"
                                and r["ever_placed"] != "1"],
        "placed, not success": [x for x, r in zip(ret, pol)
                                if r["ever_placed"] == "1"
                                and r["success_once"] != "1"],
        "success": [x for x, r in zip(ret, pol) if r["success_once"] == "1"],
    }
    return dict(n=len(pol), n1=n1, n0=n0, auc=a, z=auc_z(a, n1, n0), range=rng,
                r_pb=point_biserial(ret, suc),
                cond_auc=ca, cond_n1=cn1, cond_n0=cn0,
                cond_z=auc_z(ca, cn1, cn0), cond_label=stage["label"],
                mean_success=m1, mean_fail=m0, gap_frac=(m1 - m0) / rng if rng else 0,
                groups={k: mean(v) for k, v in groups.items()},
                group_n={k: len(v) for k, v in groups.items()},
                arms={arm: mean(_f(r, "ep_return") for r in rows if r["arm"] == arm)
                      for arm in sorted({r["arm"] for r in rows})})


def check_layer_d(out, label, mode):
    path = os.path.join(out, f"align_{label}.csv")
    if not os.path.exists(path):
        check(False, "4 alignment", f"{path} missing - run `bash t3/run.sh align`")
        return None
    rows = _rows(path)
    hdr = list(rows[0].keys()) if rows else []
    if not check(hdr == ALIGN_COLUMNS, "4 alignment  header is spec.ALIGN_COLUMNS",
                 "" if hdr == ALIGN_COLUMNS else f"got {hdr}"):
        return None

    s = align_stats(rows, mode)
    if not check(s["n"] >= ALIGN_N_MIN, "4 alignment  enough episodes",
                 f"{s['n']} policy episodes (needs {ALIGN_N_MIN}, "
                 f"{s['n1']} success / {s['n0']} fail)"):
        return s

    check(s["auc"] >= ALIGN_AUC_MIN,
          "4 alignment  ranks successes above failures",
          f"AUC {s['auc']:.3f} (needs {ALIGN_AUC_MIN}), "
          f"z {s['z']:.1f}, r_pb {s['r_pb']:.2f}")
    check(s["z"] >= ALIGN_Z_MIN,
          "4 alignment  the separation is bigger than chance",
          f"z {s['z']:.1f} against the null (needs {ALIGN_Z_MIN}); "
          f"n1={s['n1']} n0={s['n0']}")
    check(s["gap_frac"] >= ALIGN_MEAN_GAP_FRAC,
          "4 alignment  the gap is a real size, not just significant",
          f"mean(success) - mean(fail) = {s['gap_frac']:.3f} of the observed "
          f"range (needs {ALIGN_MEAN_GAP_FRAC})")

    # THE one. A grasp-farming reward scores well above and at chance here.
    if s["cond_n1"] + s["cond_n0"] < ALIGN_COND_N_MIN:
        note(f"4 alignment  {s['cond_label']}",
             f"only {s['cond_n1'] + s['cond_n0']} episodes in the stratum "
             f"(needs {ALIGN_COND_N_MIN}) - reported, not gated")
    else:
        check(s["cond_auc"] >= ALIGN_COND_AUC_MIN,
              f"4 alignment  ranks by the stage this mode fails at "
              f"({s['cond_label']})",
              f"AUC {s['cond_auc']:.3f} (needs {ALIGN_COND_AUC_MIN}), "
              f"z {s['cond_z']:.1f}, n={s['cond_n1']}/{s['cond_n0']}")

    # The binding stage test. See spec.ALIGN_STAGE_GAP_FRAC: the conditional
    # AUC above has a structural floor (the successful episodes are a subset of
    # the placed ones), so it is reported rather than relied on. This is not
    # subject to that - a reward that pays for grasping and not for placing
    # inverts the first step of this ladder however large its success bonus is.
    g, gn = s["groups"], s["group_n"]
    order = ["grasped, not placed", "placed, not success", "success"]
    have = [k for k in order if gn[k] > 0]
    vals = [g[k] for k in have]
    need = ALIGN_STAGE_GAP_FRAC * s["range"]
    steps_ok = all(b >= a + need for a, b in zip(vals, vals[1:]))
    check(steps_ok, "4 alignment  mean return rises with the stage reached",
          "  <  ".join(f"{k}={g[k]:.1f}(n={gn[k]})" for k in have)
          + f"   [each step needs +{need:.1f}]")

    for arm in ("jitter", "zero"):
        if arm not in s["arms"]:
            note(f"4 alignment  vs {arm}", "arm not run")
            continue
        p, c = s["arms"]["policy"], s["arms"][arm]
        ok = c <= 0 or p / c >= ALIGN_CONTROL_RATIO if c > 0 else p > c
        check(ok, f"4 alignment  the real policy beats `{arm}`",
              f"{p:.1f} vs {c:.1f}" +
              (f" ({p / c:.2f}x, needs {ALIGN_CONTROL_RATIO}x)" if c > 0 else ""))
    return s


# ---------------------------------------------------------------------------
# 5 - the join back onto T-II, and manifest consistency
# ---------------------------------------------------------------------------


def check_join(out, label, mode, t2_results):
    """The align rows' initial states must be T-II's, to 1e-5.

    Free, and it is the structural check on the whole subclass: it proves
    StackCube-T3-v1 with the sampler off reproduces StackCube-v1's
    initial-state distribution exactly, i.e. that nothing about the reward
    plumbing perturbed what T-IV will be scored on.
    """
    ap = os.path.join(out, f"align_{label}.csv")
    tp = os.path.join(t2_results, f"mode_{mode}_seed1.csv")
    if not (os.path.exists(ap) and os.path.exists(tp)):
        note("5 join", "align or T-II CSV missing, skipping")
        return
    t2 = {int(r["seed"]): r for r in _rows(tp)}
    cols = ["cubeA_x", "cubeA_y", "cubeA_theta", "cubeB_x", "cubeB_y", "cubeB_theta"]
    bad, n = [], 0
    for r in _rows(ap):
        s = int(r["seed"])
        if s not in t2:
            bad.append((s, "not a T-II seed"))
            continue
        n += 1
        for c in cols:
            if abs(_f(r, c) - float(t2[s][c])) > 1e-5:
                bad.append((s, f"{c} {_f(r, c):.6f} vs {float(t2[s][c]):.6f}"))
                break
    check(not bad, "5 join  initial states match T-II's record of these seeds",
          f"{n} episodes joined on all 6 pose columns" if not bad
          else f"{len(bad)} mismatches, first: {bad[0]}")


def check_manifests(out, label, run):
    path = os.path.join(out, f"align_{label}_manifest.json")
    if not os.path.exists(path):
        note("5 manifest", f"{path} missing")
        return
    m = json.load(open(path))
    check(int(m.get("t3_sampler", -1)) == 0,
          "5 manifest  the biased sampler was OFF for the measurement",
          "t3_sampler=0" if int(m.get("t3_sampler", -1)) == 0 else
          f"t3_sampler={m.get('t3_sampler')} - these episodes are NOT the T-II "
          f"seeds' states, so the AUC is over the wrong population")
    ck = {json.load(open(p)).get("ckpt_sha256")
          for p in glob.glob(os.path.join(out, "align_*_manifest.json"))}
    check(len(ck) <= 1, "5 manifest  one checkpoint behind every arm",
          f"sha256 {list(ck)[0] if ck else '?'}" if len(ck) <= 1
          else f"{len(ck)} different checkpoints: {ck}")

    if run and os.path.exists(os.path.join(run, REWARD_FILE)):
        from loader import check_static  # noqa: F401 - keeps import local
        import ast
        src = open(os.path.join(run, REWARD_FILE)).read()
        want = None
        for node in ast.parse(src).body:
            if isinstance(node, ast.Assign) and any(
                    getattr(t, "id", "") == "REWARD_MAX" for t in node.targets):
                want = node.value.value
        check(want is not None and abs(float(m.get("reward_max", -1)) - want) < 1e-9,
              "5 manifest  REWARD_MAX matches the file that was measured",
              f"{m.get('reward_max')} == {want}")


# ---------------------------------------------------------------------------
# 6 - layer E
# ---------------------------------------------------------------------------


def check_layer_e(out, mode):
    path = os.path.join(out, f"sampler_{mode}.json")
    if not os.path.exists(path):
        check(False, "6 sampler", f"{path} missing - run `bash t3/run.sh sampler`")
        return
    m = json.load(open(path))
    n = m["draws"]

    for key, lab in (("quat_bad", "quaternions are unit and about z only"),
                     ("z_bad", "cubes sit on the table"),
                     ("separation_bad", "cubes do not interpenetrate"),
                     ("support_bad", "inside the nominal support"),
                     ("reach_bad", "inside the arm's reach")):
        check(m[key] == 0, f"6 sampler  {lab}",
              "" if m[key] == 0 else f"{m[key]} of {n} draws violate it")

    check(m["hit_rate"] >= SAMPLER_HIT_RATE_MIN,
          "6 sampler  draws land in the failure region",
          f"{m['hit_rate']:.3f} (needs {SAMPLER_HIT_RATE_MIN})")
    if m.get("enrichment"):
        check(m["enrichment"] >= SAMPLER_ENRICHMENT_MIN,
              "6 sampler  biased well above the nominal rate",
              f"{m['enrichment']:.1f}x the environment's own "
              f"{m['base_rate']:.4f} (needs {SAMPLER_ENRICHMENT_MIN}x)")
    else:
        note("6 sampler  enrichment", "no seed index, base rate unknown")

    for key in ("sd_cubeA_x", "sd_cubeA_y", "sd_cubeB_x", "sd_cubeB_y"):
        check(m[key] >= SAMPLER_SD_MIN_XY, f"6 sampler  spread: {key}",
              f"{m[key]:.4f} m (needs {SAMPLER_SD_MIN_XY})")
    check(m["sd_face_gap"] >= SAMPLER_SD_MIN_GAP, "6 sampler  spread: face_gap",
          f"{m['sd_face_gap']:.4f} m (needs {SAMPLER_SD_MIN_GAP})")
    check(min(m["yaw_bins_A"], m["yaw_bins_B"]) >= SAMPLER_YAW_BINS_MIN,
          "6 sampler  yaw is covered, not fixed",
          f"{m['yaw_bins_A']}/{m['yaw_bins_B']} of {SAMPLER_YAW_BINS} bins "
          f"(needs {SAMPLER_YAW_BINS_MIN})")
    check(m["distinct_frac"] >= SAMPLER_DISTINCT_FRAC,
          "6 sampler  draws are distinct",
          f"{m['distinct_frac']:.2f} distinct at 1 mm "
          f"(needs {SAMPLER_DISTINCT_FRAC})")
    check(bool(m.get("deterministic")),
          "6 sampler  same torch seed gives the same draws",
          "" if m.get("deterministic") else
          "NOT deterministic - it is drawing from something other than torch, "
          "which breaks reset(seed=s) and every seed-addressed evaluation")

    if m.get("eval_seed_coverage") is not None:
        check(m["eval_seed_coverage"] >= SAMPLER_COVER_FRAC,
              "6 sampler  covers the episodes T-IV is scored on",
              f"{m['eval_seed_coverage']:.2f} of {m['eval_seed_n']} "
              f"(needs {SAMPLER_COVER_FRAC})")
    if m.get("env_determinism") is None:
        note("6 sampler  reset(seed=s) through the real env",
             m.get("env_determinism_error", "not run"))
    else:
        check(m["env_determinism"],
              "6 sampler  reset(seed=s) is reproducible through the env")


# ---------------------------------------------------------------------------


def calibration(out, mode, gen):
    """The stock reward's numbers beside the generated one's, if it was run.

    Without this column an AUC is an uncalibrated number. With it, the report
    can say whether the LLM beat the reward it was shown - and both answers are
    findings.
    """
    path = os.path.join(out, "align_stock.csv")
    if not os.path.exists(path) or gen is None:
        note("calibration", "stock-reward arm not run "
                            "(`bash t3/run.sh calibrate`) - every AUC below is "
                            "uncalibrated")
        return
    s = align_stats(_rows(path), mode)
    print(f"\n  calibration against ManiSkill's own 8-stage dense reward, "
          f"same {s['n']} episodes")
    print(f"    {'':<22} {'generated':>10} {'stock':>10}")
    print(f"    {'AUC (success)':<22} {gen['auc']:>10.3f} {s['auc']:>10.3f}")
    print(f"    {'AUC (' + gen['cond_label'] + ')':<22} "
          f"{gen['cond_auc']:>10.3f} {s['cond_auc']:>10.3f}")
    print(f"    {'r_pb':<22} {gen['r_pb']:>10.2f} {s['r_pb']:>10.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--mode", required=True)
    ap.add_argument("--label")
    ap.add_argument("--run")
    ap.add_argument("--t2-results",
                    default=os.environ.get(
                        "T2_RESULTS",
                        os.path.join(os.path.dirname(HERE), "t2", "results")))
    a = ap.parse_args()
    label = a.label or a.mode
    print(f"verifying {a.out}  (mode '{a.mode}', label '{label}')\n")

    check_static_artifacts(a.run)
    pj = check_layer_b(a.out, label)
    reward_max = pj["reward_max"] if pj else 8.0
    check_layer_c(a.out, label, reward_max, (pj or {}).get("notes", {}))
    gen = check_layer_d(a.out, label, a.mode)
    check_join(a.out, label, a.mode, a.t2_results)
    check_manifests(a.out, label, a.run)
    check_layer_e(a.out, a.mode)
    calibration(a.out, a.mode, gen)

    print()
    if FAILS:
        print(f"{len(FAILS)} CHECK(S) FAILED:")
        for f in FAILS:
            print(f"    {f}")
        print(f"\nThis generation is NOT usable for T-IV. Keep it - a rejected\n"
              f"generation is evidence, and which check caught it is the report's\n"
              f"account of how LLM-written rewards fail. Tune\n"
              f"t3/prompts/hacking.md and regenerate into a new directory.")
        sys.exit(1)
    print("every check passed - this reward and sampler are usable for T-IV.")


if __name__ == "__main__":
    main()
