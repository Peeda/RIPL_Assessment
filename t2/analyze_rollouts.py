#!/usr/bin/env python
"""Turn mined rollouts into the T-II figures, tables and video shortlist.

    python t2/analyze_rollouts.py mined.csv [more.csv ...] [--figdir figures]

Produces:
  - success rate vs cube separation, with Wilson intervals
  - the failure taxonomy, which is the part T-II is actually graded on
  - a 2-D scatter over (separation, relative_yaw_mod90)
  - the success_once / success_at_end gap
  - a --seeds line to paste into record_seeds.py, chosen by rule

Wilson rather than the sqrt(p(1-p)/n) CLAUDE.md quotes: at n~100 the interesting
bins sit near p=0, where the normal interval runs below zero and understates the
uncertainty. Wilson stays inside [0,1] and does not degenerate when a bin goes
0-for-20. Same data, honest bars.
"""
import csv
import math
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t2_common import PANDA_BASE_XY, geom_from_row, wilson  # noqa: E402

FIGDIR = sys.argv[sys.argv.index("--figdir") + 1] if "--figdir" in sys.argv else "figures"
CSVS = [a for a in sys.argv[1:] if a.endswith(".csv")]


def load(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            for r in csv.DictReader(f):
                for k, v in list(r.items()):
                    if v == "":
                        r[k] = None
                        continue
                    try:
                        r[k] = float(v)
                    except ValueError:
                        pass
                # face_gap and the reach distances are derived, not logged, so
                # the CSVs mined before they existed gain them here. Every
                # input they need is already a column.
                try:
                    r.update(geom_from_row(r))
                except (KeyError, TypeError, ValueError):
                    pass
                rows.append(r)
    return rows


def taxonomy(r):
    """Which of the four ways an episode can end. The three flags come free from
    evaluate()'s info dict, and they turn a binary failure into a real taxonomy -
    two 'failure modes' differing in which flag is false are different modes."""
    if r.get("success_once") == 1:
        return "success"
    if not r.get("ever_is_cubeA_grasped"):
        return "never grasped"
    if not r.get("ever_is_cubeA_on_cubeB"):
        return "grasped, never placed"
    if r.get("is_cubeA_on_cubeB") != 1:
        return "placed, then toppled"
    return "placed, never released"


def targeted(r):
    """True if the episode came from a targeted pass rather than the nominal
    one. A targeted pass oversamples its region by construction, so it must
    never contribute to anything that reads as a DENSITY - only to rates
    computed within a bin, where both passes draw from the same conditional."""
    run = str(r.get("run_id", ""))
    return "region" in run or "targeted" in run


def table(title, groups):
    print(f"\n{title}")
    print(f"  {'bin':>16}  {'n':>5} {'seeds':>6}  {'succ':>6}  "
          f"{'95% Wilson':>16}  taxonomy")
    for name, rs in groups:
        n = len(rs)
        if not n:
            continue
        k = sum(1 for r in rs if r.get("success_once") == 1)
        ns = len({r.get("seed") for r in rs})
        lo, hi = wilson(k, n)
        tax = defaultdict(int)
        for r in rs:
            tax[taxonomy(r)] += 1
        top = sorted(tax.items(), key=lambda x: -x[1])[:2]
        desc = ", ".join(f"{a} {b/n:.0%}" for a, b in top)
        print(f"  {name:>16}  {n:5d} {ns:6d}  {k/n:6.3f}  "
              f"[{lo:.3f}, {hi:.3f}]  {desc}")


def curve(ax, groups, key, xlabel, title, marks=(), unit=1000):
    """One success-vs-feature panel with Wilson bars sized by DISTINCT SEEDS.

    Sized by seeds rather than episodes because the targeted passes run each
    seed more than once: episodes overstate the information in a bin the
    repeats dominate. The conservative reading, and the one a region claim has
    to survive.
    """
    xs, ys, els, ehs, ns, nss = [], [], [], [], [], []
    for _, rs in groups:
        if not rs:
            continue
        x = np.mean([r[key] for r in rs]) * unit
        n, nseed = len(rs), len({r.get("seed") for r in rs})
        p = sum(1 for r in rs if r.get("success_once") == 1) / n
        lo, hi = wilson(round(p * nseed), nseed)
        xs.append(x); ys.append(p)
        els.append(p - lo); ehs.append(hi - p); ns.append(n); nss.append(nseed)
    ax.errorbar(xs, ys, yerr=[els, ehs], marker="o", capsize=3, zorder=3)
    # Counts in a fixed row along the bottom, rotated, rather than above each
    # whisker: anchoring them to the data made adjacent bins collide wherever
    # two points sat close in y, which is exactly what the flat middle does.
    for x, m, msd in zip(xs, ns, nss):
        ax.annotate(f"{m}" if m == msd else f"{m} ({msd})", (x, 0.03),
                    ha="center", va="bottom", rotation=90, fontsize=6.5,
                    color="0.35")
    for x, c, lab in marks:
        ax.axvline(x, ls="--", c=c, lw=1)
        ax.text(x, 0.99, " " + lab, transform=ax.get_xaxis_transform(),
                fontsize=7, color=c, va="top",
                ha="left" if x < np.mean(xs) else "right")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("success_once")
    ax.set_title(title, fontsize=10)
    ax.set_ylim(-0.02, 1.08)
    ax.grid(alpha=0.3)


def main():
    if not CSVS:
        sys.exit("usage: t2/analyze_rollouts.py mined.csv [more.csv ...]")
    rows = load(CSVS)
    rows = [r for r in rows if r.get("success_once") is not None]
    print(f"loaded {len(rows)} episodes from {len(CSVS)} file(s)")

    # Report per pass, never pooled. A targeted pass oversamples the failure
    # region by construction, so pooling it with the nominal pass yields a
    # number that is neither the unconditional success rate nor a conditional
    # one. Only the nominal pass estimates P(success).
    by_run = defaultdict(list)
    for r in rows:
        by_run[r.get("run_id", "?")].append(r)
    print("\n  per pass (NOT pooled - a targeted pass is not a sample of the "
          "nominal distribution):")
    for run, rs in sorted(by_run.items()):
        n = len(rs)
        k = sum(1 for r in rs if r.get("success_once") == 1)
        ke = sum(1 for r in rs if r.get("success_at_end") == 1)
        lo, hi = wilson(k, n)
        nseed = len({r.get("seed") for r in rs})
        note = "  <- conditional on the region, not P(success)" \
               if targeted(rs[0]) else \
               "  <- the unconditional number, comparable to T-I"
        print(f"    {run:>12}  n={n:<5d} seeds={nseed:<5d} "
              f"success_once {k/n:.3f} [{lo:.3f}, {hi:.3f}]{note}")
        print(f"    {'':>12}  success_at_end {ke/n:.3f}"
              + (f"   gap {(k-ke)/n:+.3f}: placed, then toppled before step 200 - "
                 f"a distinct mode from any separation effect"
                 if (k - ke) / n > 0.05 else ""))
        if nseed < n:
            print(f"    {'':>12}  NOTE {n} episodes over {nseed} distinct states "
                  f"({n/nseed:.1f} repeats each). Wilson assumes independent")
            print(f"    {'':>12}       draws, so the interval above is optimistic; "
                  f"the honest n for a region claim is {nseed}.")

    # The far tail is failure mode 2, so it gets its own bins instead of being
    # lumped into one '>200 mm'. 0.0585 is the sampler's enforced floor.
    edges = [0.0585, 0.070, 0.080, 0.100, 0.140, 0.200, 0.260, 0.320, 1.0]
    groups = []
    for a, b in zip(edges, edges[1:]):
        rs = [r for r in rows if a <= (r.get("separation") or -1) < b]
        groups.append((f"{a*1000:.0f}-{b*1000:.0f} mm" if b < 1 else f">{a*1000:.0f} mm", rs))
    # Binning IS legitimate to pool: within a separation bin, nominal-drawn and
    # region-drawn episodes are both draws from the same conditional
    # distribution. Only the unconditional headline above must stay separated.
    table("success_once vs initial cube separation (passes pooled - valid "
          "within a bin)", groups)

    # ---- the refined axes -------------------------------------------------
    # separation is the axis T-II started on, and it is the wrong one twice
    # over. What follows replaces it; the separation table above stays so the
    # report can show the before as well as the after.
    def bins(key, edges, unit=1000, fmt="{:.0f}", where=None):
        src = [r for r in rows if where is None or where(r)]
        out = []
        for a, b in zip(edges, edges[1:]):
            rs = [r for r in src if r.get(key) is not None and a <= r[key] < b]
            out.append((f"{fmt.format(a*unit)}-{fmt.format(b*unit)}", rs))
        return out

    # face_gap: the free space between the cube FACES, which is what separation
    # discards. Both cube yaws enter through the A->B bearing, which is why
    # relative_yaw on its own (below) measures nothing.
    table("success_once vs FACE GAP, mm - the clearance the gripper actually "
          "has\n  (separation minus both cubes' half-extents along the A->B "
          "bearing)",
          bins("face_gap", [-0.02, 0.005, 0.02, 0.04, 0.07, 0.12, 0.20, 1.0]))

    # dist_max: distance from the Panda's base to the FARTHER cube. This is an
    # inverted U, not a trend - both tails fail and the middle is best - so it
    # is binned rather than fitted, and both ends get their own bins.
    table("success_once vs DISTANCE FROM PANDA BASE to the farther cube, mm\n"
          "  (base at %.3f, %.3f - not the world origin)" % PANDA_BASE_XY,
          bins("dist_max", [0.40, 0.52, 0.60, 0.68, 0.76, 1.0]))
    table("success_once vs distance to the NEARER cube, mm (the near-base tail)",
          bins("dist_min", [0.40, 0.48, 0.52, 0.58, 0.64, 1.0]))

    # Are these two effects the same effect? The 2x2 says no: reach costs about
    # twice what the separation tail does, and they stack.
    nom_rows = [r for r in rows if not targeted(r)]
    if nom_rows:
        print("\n  separation tail x reach tail (NOMINAL pass only, n=%d)"
              % len(nom_rows))
        print(f"    {'':22} {'dist_max < 0.76':>16} {'dist_max >= 0.76':>17}")
        for slab, sf in (("separation < 260 mm", lambda r: r["separation"] < 0.26),
                         ("separation >= 260 mm", lambda r: r["separation"] >= 0.26)):
            cells = []
            for df in (lambda r: r["dist_max"] < 0.76, lambda r: r["dist_max"] >= 0.76):
                ch = [r for r in nom_rows if sf(r) and df(r)]
                cells.append(f"{sum(1 for r in ch if r.get('success_once') == 1)/len(ch):.3f} "
                             f"(n={len(ch)})" if ch else "-")
            print(f"    {slab:22} {cells[0]:>16} {cells[1]:>17}")
        print("    If reach were just separation in disguise, the second column")
        print("    would be flat. It is not - the two effects stack.")

        # The mechanism claim: the far mode fails DIFFERENTLY depending on which
        # cube is out of reach. A far -> cannot grasp it. B far -> grasps fine,
        # cannot place. Those are two different failures, not one.
        far = [r for r in nom_rows if r["dist_max"] >= 0.76]
        near = [r for r in nom_rows if r["dist_min"] < 0.52 and r["dist_max"] < 0.76]
        mid = [r for r in nom_rows if r["dist_min"] >= 0.52 and r["dist_max"] < 0.76]
        print("\n  where each reach band breaks (NOMINAL only) - the mechanism claim")
        print(f"    {'band':>26} {'n':>5} {'succ':>6} {'never grasped':>14} "
              f"{'grasped, no place':>18}")
        def line(lab, ch):
            if not ch:
                return
            n = len(ch)
            ng = sum(1 for r in ch if not r.get("ever_is_cubeA_grasped")) / n
            gp = sum(1 for r in ch if r.get("ever_is_cubeA_grasped")
                     and not r.get("ever_is_cubeA_on_cubeB")) / n
            sc = sum(1 for r in ch if r.get("success_once") == 1) / n
            print(f"    {lab:>26} {n:5d} {sc:6.3f} {ng:14.3f} {gp:18.3f}")
        line("dist_min < 520 mm", near)
        line("mid band", mid)
        # far_is_B only says which cube is FARTHER, so splitting on it alone
        # puts every both-cubes-far episode into one side or the other and the
        # cell measures "how many cubes are far" rather than "which one". The
        # conditioned cells below are the honest split: vary one cube's
        # distance with the other held comfortable.
        line("only cubeA far (B<720)",
             [r for r in nom_rows if r["dist_A"] >= 0.76 and r["dist_B"] < 0.72])
        line("only cubeB far (A<720)",
             [r for r in nom_rows if r["dist_B"] >= 0.76 and r["dist_A"] < 0.72])
        line("BOTH >= 720 mm", [r for r in nom_rows if r["dist_min"] >= 0.72])
        line("  (unconditioned far=A)", [r for r in far if not r["far_is_B"]])
        line("  (unconditioned far=B)", [r for r in far if r["far_is_B"]])
        print("    Near-base grasps normally and fails at the PLACE step.")
        print("    Only-cubeB-far grasps EVERY time - it is not a reach failure;")
        print("    see the hold-given-place column below. Grasping only breaks")
        print("    down when BOTH cubes are far, which is where the IK saturates")
        print("    and where no bounded residual can help.")
        print("    The two unconditioned rows are shown for contrast: they are")
        print("    contaminated by both-far episodes and overstate far=A.")

        # place-stage breakdown, which is what separates the two T-IV targets
        print("\n  where each region loses episodes, by stage (NOMINAL only)")
        print(f"    {'region':>30} {'n':>5} {'grasp':>7} {'place|grasp':>12} "
              f"{'hold|place':>11}")
        def stages(lab, ch):
            if len(ch) < 8:
                return
            g = [r for r in ch if r.get("ever_is_cubeA_grasped")]
            o = [r for r in ch if r.get("ever_is_cubeA_on_cubeB")]
            print(f"    {lab:>30} {len(ch):5d} "
                  f"{sum(1 for r in ch if r.get('ever_is_cubeA_grasped'))/len(ch):7.3f} "
                  f"{sum(1 for r in g if r.get('ever_is_cubeA_on_cubeB'))/len(g):12.3f} "
                  f"{sum(1 for r in o if r.get('success_once') == 1)/len(o):11.3f}")
        stages("reference: both<720, gap>50",
               [r for r in nom_rows if r["dist_max"] < 0.72 and r["face_gap"] >= 0.05])
        stages("target A: face_gap < 25 mm",
               [r for r in nom_rows if r["face_gap"] < 0.025])
        stages("target B: cubeB far, cubeA close",
               [r for r in nom_rows if r["dist_B"] >= 0.76 and r["dist_A"] < 0.72
                and r["face_gap"] >= 0.05])
        print("    Target A fails BEFORE the stack exists; target B builds one")
        print("    and cannot make it stay. Different stages, different modes -")
        print("    which is the case for taking both to T-IV.")

        # Toppling: the largest single failure class, and NOT a region of the
        # initial-state distribution - no initial-state feature predicts it.
        # What does is how far cubeB got shoved, which is an outcome. So it is
        # the downstream consequence of the face-gap mode, not a peer to it.
        pl = [r for r in nom_rows if r.get("ever_is_cubeA_on_cubeB")
              and r.get("cubeB_displacement") is not None]
        if pl:
            print(f"\n  of {len(pl)} episodes that got A onto B, "
                  f"{sum(1 for r in pl if r.get('success_at_end') == 1)/len(pl):.3f} "
                  f"still held at step 200.")
            print("    hold rate by how far cubeB moved:")
            # Fixed thresholds, not quintiles. The displacement distribution is
            # a spike at zero with a long tail, so quintiles put three cut
            # points inside the spike and report three bins that all round to
            # 0.0 mm - which reads as noise when the effect is real.
            de = [0.0, 0.0001, 0.001, 0.005, 0.020, 9.0]
            for a, b in zip(de, de[1:]):
                ch = [r for r in pl if a <= r["cubeB_displacement"] < b]
                if not ch:
                    continue
                print(f"      {a*1000:5.1f}-{b*1000:5.1f} mm  n={len(ch):4d}  "
                      f"hold {sum(1 for r in ch if r.get('success_at_end') == 1)/len(ch):.3f}")
            print("    Toppling tracks how far cubeB was shoved, not the initial")
            print("    state - so it is the face-gap mode's consequence, not a mode.")

    q = [r for r in rows if r.get("relative_yaw_mod90") is not None]
    if q:
        ye = [-math.pi/4, -math.pi/8, 0, math.pi/8, math.pi/4]
        table("success_once vs relative yaw (mod 90 deg, the physical axis)",
              [(f"{a*180/math.pi:+.0f}..{b*180/math.pi:+.0f}",
                [r for r in q if a <= r["relative_yaw_mod90"] < b])
               for a, b in zip(ye, ye[1:])])

    # cubeB displacement: the mechanism behind the close-separation hypothesis.
    close = [r for r in rows if (r.get("separation") or 1) < 0.080]
    far = [r for r in rows if (r.get("separation") or 0) >= 0.140]
    def disp(rs):
        v = [r["cubeB_displacement"] for r in rs if r.get("cubeB_displacement") is not None]
        return np.mean(v) * 1000 if v else float("nan")
    if close and far:
        print(f"\n  cubeB displaced during the episode:")
        print(f"    sep < 80 mm : {disp(close):6.1f} mm  (n={len(close)})")
        print(f"    sep >=140 mm: {disp(far):6.1f} mm  (n={len(far)})")
        print(f"  If the close number is much larger, the approach onto A is fouling B -")
        print(f"  that is the mechanism, not just a correlation with separation.")

    # --- video shortlist, chosen by rule rather than by eye ------------------
    by_class = defaultdict(list)
    for r in rows:
        by_class[taxonomy(r)].append(r)
    picks = []
    for cls, rs in sorted(by_class.items()):
        rs = [r for r in rs if r.get("separation") is not None]
        if not rs:
            continue
        med = np.median([r["separation"] for r in rs])
        rs.sort(key=lambda r: abs(r["separation"] - med))
        picks += [(cls, int(r["seed"])) for r in rs[:3]]
    print(f"\n  video shortlist - 3 per class, nearest that class's median separation:")
    for cls, s in picks:
        print(f"    {cls:>22}  seed {s}")
    print(f"\n    python t2/record_seeds.py CKPT --want fail --seeds "
          f"{','.join(str(s) for c, s in picks if c != 'success')}")

    # --- figures -------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  (matplotlib missing - tables only)")
        return
    os.makedirs(FIGDIR, exist_ok=True)

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
    xs, ys, els, ehs, ns, nss = [], [], [], [], [], []
    for name, rs in groups:
        if not rs:
            continue
        sep = np.mean([r["separation"] for r in rs]) * 1000
        kk = sum(1 for r in rs if r.get("success_once") == 1)
        n, nseed = len(rs), len({r.get("seed") for r in rs})
        p = kk / n
        # Wilson assumes independent draws. The targeted passes run each seed
        # more than once, so episodes overstate the information in a bin the
        # repeats dominate. Size the interval by DISTINCT SEEDS instead - the
        # conservative reading, and the one a region claim has to survive.
        l, h = wilson(round(p * nseed), nseed)
        xs.append(sep); ys.append(p)
        els.append(p - l); ehs.append(h - p); ns.append(n); nss.append(nseed)
    ax[0].errorbar(xs, ys, yerr=[els, ehs], marker="o", capsize=3, zorder=3)
    # Counts go in a fixed row along the bottom rather than above each whisker.
    # Anchoring them to the data made adjacent bins collide whenever two points
    # sat close in y - which is exactly what happens across the flat middle.
    # Vertical, so the three narrow bins below 100 mm cannot overlap each other
    # however tightly they bunch on the x axis.
    for x, m, msd in zip(xs, ns, nss):
        lab = f"{m}" if m == msd else f"{m} ({msd})"
        ax[0].annotate(lab, (x, 0.03), ha="center", va="bottom", rotation=90,
                       fontsize=6.5, color="0.35")
    for x, c, lab in ((80, "crimson", "mode 1: <80 mm\n(pre-registered)"),
                      (260, "tab:purple", "mode 2: >260 mm\n(exploratory)")):
        ax[0].axvline(x, ls="--", c=c, lw=1)
        # axes-fraction y keeps the label clear of the curve at any y-range
        ax[0].text(x, 0.99, " " + lab, transform=ax[0].get_xaxis_transform(),
                   fontsize=7, color=c, va="top",
                   ha="left" if x < 200 else "right")
    ax[0].set_xlabel("initial cube separation (mm)\n"
                     "bin labels: episodes (distinct seeds, where repeated)",
                     fontsize=9)
    ax[0].set_ylabel("success_once")
    ax[0].set_title("Success vs separation (Wilson 95%, sized by distinct seeds)")
    ax[0].set_ylim(-0.02, 1.08)
    ax[0].grid(alpha=0.3)

    # NOMINAL EPISODES ONLY. This panel reads as a density - how often each
    # part of the initial-state plane occurs - and the targeted passes
    # oversample their regions by construction. Pooling them here made the
    # sub-80 mm stripe look like ~30% of the distribution when it is 7.7%,
    # which is the one number a reader takes straight off this plot.
    nom = [r for r in rows if not targeted(r)
           and r.get("relative_yaw_mod90") is not None]
    ok = [r for r in nom if r.get("success_once") == 1]
    no = [r for r in nom if r.get("success_once") != 1]
    for rs, c, lab in ((ok, "tab:green", "success"), (no, "tab:red", "failure")):
        ax[1].scatter([r["separation"] * 1000 for r in rs],
                      [r["relative_yaw_mod90"] * 180 / math.pi for r in rs],
                      s=8, alpha=0.5, c=c, label=f"{lab} ({len(rs)})")
    for x, c in ((80, "crimson"), (260, "tab:purple")):
        ax[1].axvline(x, ls="--", c=c, lw=1)
    ax[1].set_xlabel("initial cube separation (mm)")
    ax[1].set_ylabel("relative yaw mod 90 (deg)")
    n_reg = len(rows) - len(nom)
    ax[1].set_title(f"Initial-state plane - NOMINAL pass only (n={len(nom)})")
    if n_reg:
        ax[1].annotate(f"{n_reg} targeted episodes excluded:\nthey oversample "
                       f"their region by design",
                       (0.98, 0.02), xycoords="axes fraction", ha="right",
                       va="bottom", fontsize=6.5, color="0.35", style="italic")
    ax[1].legend(fontsize=8, loc="upper right")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/t2_separation.png", dpi=150)

    # Grouped by pass, because a pooled taxonomy is a mixture over whatever
    # ratio of nominal to targeted episodes happened to be mined - not a
    # property of anything. Side by side it answers the question that matters:
    # does the region fail DIFFERENTLY, or just more?
    cls = sorted(by_class, key=lambda c: -len(by_class[c]))
    passes = sorted(by_run, key=lambda r: (targeted(by_run[r][0]), r))
    fig2, ax2 = plt.subplots(figsize=(8, 0.55 * len(cls) * len(passes) + 1.4))
    hgt = 0.8 / len(passes)
    ypos = np.arange(len(cls))
    for i, run in enumerate(passes):
        rs = by_run[run]
        fr = [sum(1 for r in rs if taxonomy(r) == c) / len(rs) for c in cls]
        ax2.barh(ypos + i * hgt, fr, height=hgt,
                 label=f"{os.path.basename(str(run))} (n={len(rs)})")
    ax2.set_yticks(ypos + 0.4 - hgt / 2)
    ax2.set_yticklabels(cls)
    ax2.invert_yaxis()
    ax2.set_xlabel("fraction of that pass's episodes")
    ax2.set_title("Outcome taxonomy, per pass")
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3, axis="x")
    fig2.tight_layout()
    fig2.savefig(f"{FIGDIR}/t2_taxonomy.png", dpi=150)

    # --- the refined axes ---------------------------------------------------
    # This is the figure the report leads with. t2_separation.png stays as the
    # "before": it is the axis T-II started on and it is the wrong one.
    fig3, ax3 = plt.subplots(2, 2, figsize=(12.5, 8.4))

    # TOP ROW - the two T-IV targets, on the axes their regions are defined on.
    curve(ax3[0][0],
          bins("face_gap", [-0.02, 0.005, 0.02, 0.04, 0.07, 0.12, 0.20, 1.0]),
          "face_gap",
          "clearance between cube faces (mm)\n"
          "bin labels: episodes (distinct seeds, where repeated)",
          "TARGET A - success vs face gap",
          marks=[(25, "crimson", "region: <25 mm")])

    # Target B is dist_B CONDITIONED on cubeA being comfortable. Plotting raw
    # dist_max here would be a different region: it mixes in both-cubes-far
    # episodes, which is the corner the region deliberately excludes because
    # the IK saturates there and no bounded residual can help.
    curve(ax3[0][1],
          bins("dist_B", [0.45, 0.60, 0.68, 0.72, 0.76, 0.80, 1.0],
               where=lambda r: r.get("dist_A") is not None and r["dist_A"] < 0.72),
          "dist_B",
          "distance from Panda base to cubeB (mm)\n"
          "cubeA held < 720 mm, so only cubeB's distance varies",
          "TARGET B - success vs cubeB distance",
          marks=[(760, "tab:purple", "region: >=760 mm")])

    # BOTTOM ROW - context, not targets.
    curve(ax3[1][0], bins("dist_max", [0.40, 0.52, 0.60, 0.68, 0.76, 1.0]),
          "dist_max",
          "distance from Panda base to the farther cube (mm)",
          "context: the reach axis is an inverted U",
          marks=[(760, "tab:purple", "planner failures rise")])
    curve(ax3[1][1], bins("dist_min", [0.40, 0.48, 0.52, 0.58, 0.64, 1.0]),
          "dist_min",
          "distance from Panda base to the nearer cube (mm)",
          "third finding (not a T-IV target): near-base",
          marks=[(520, "darkorange", "region: <520 mm")])
    fig3.suptitle("The failure modes on the axes their regions are defined on",
                  fontsize=12)
    fig3.tight_layout()
    fig3.savefig(f"{FIGDIR}/t2_refined.png", dpi=150)

    # Where the two modes live on the table. NOMINAL ONLY - this reads as a
    # density and the targeted passes oversample their regions by design.
    fig4, ax4 = plt.subplots(figsize=(7.4, 5.6))
    nomr = [r for r in rows if not targeted(r) and r.get("dist_max") is not None]
    for rs, c, lab in (([r for r in nomr if r.get("success_once") == 1], "tab:green", "success"),
                       ([r for r in nomr if r.get("success_once") != 1], "tab:red", "failure")):
        ax4.scatter([r["cubeA_x"] for r in rs], [r["cubeA_y"] for r in rs],
                    s=8, alpha=0.5, c=c, label=f"cubeA, {lab} ({len(rs)})")
    bx, by = PANDA_BASE_XY
    ax4.plot([bx], [by], marker="s", ms=9, c="k", zorder=5)
    ax4.annotate("Panda base", (bx, by), textcoords="offset points",
                 xytext=(6, 6), fontsize=8)
    # Label each ring where it actually crosses the cube region, not at 45deg
    # on the circle - most of that circle is empty table the cubes never reach.
    # Staggered in y so the two labels cannot collide - at equal height the
    # arcs are only ~250 mm apart here and the text overlaps.
    for rad, c, lab, ylab in ((0.52, "darkorange", "520 mm\nnear-base inside", 0.30),
                              (0.76, "tab:purple", "760 mm\nreach edge", 0.11)):
        ax4.add_patch(plt.Circle((bx, by), rad, fill=False, ls="--", ec=c, lw=1.2))
        ax4.annotate(lab, (bx + math.sqrt(max(rad ** 2 - ylab ** 2, 0)), ylab),
                     fontsize=7, color=c, ha="center", va="bottom",
                     bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.0))
    ax4.set_xlabel("cubeA x (m)")
    ax4.set_ylabel("cubeA y (m)")
    ax4.set_title("Where the modes sit on the table (nominal pass)", fontsize=10)
    ax4.set_aspect("equal")
    # Crop to the base plus the sampler's support. The rings run off the top
    # and bottom, which is fine - only the arcs crossing the cube region carry
    # information, and showing the whole circle shrinks the data to a smudge.
    ax4.set_xlim(bx - 0.06, 0.26)
    ax4.set_ylim(-0.36, 0.40)
    ax4.legend(fontsize=7, loc="upper left")
    ax4.grid(alpha=0.3)
    fig4.tight_layout()
    fig4.savefig(f"{FIGDIR}/t2_reach_plane.png", dpi=150)

    print(f"\n  wrote {FIGDIR}/: t2_separation.png t2_taxonomy.png "
          f"t2_refined.png t2_reach_plane.png")


if __name__ == "__main__":
    main()
