#!/usr/bin/env python
"""The T-II tables and figures.

    python t2/report.py [--eval DIR] [--discovery nominal.csv] [--figdir figures]

Two inputs, two jobs:

  --eval DIR        the per-mode evaluation (mode_<tag>_seed<b>.csv), which is
                    the DELIVERABLE: per-failure-mode success over 100 rollouts
                    x 3 seeds. Produces the headline table and t2_modes.png.
  --discovery CSV   the 1,200-episode held-out nominal pass, which is the
                    EVIDENCE that the modes are regions of the initial-state
                    distribution rather than post-hoc labels. Produces
                    t2_axes.png.

Plus t2_face_gap.png, which needs no data at all - it is the diagram of what
face_gap means, and it is the fastest way to explain the axis both modes are
defined on.

Needs numpy and matplotlib. On the laptop:
    nix-shell -p "python3.withPackages(ps: [ps.numpy ps.matplotlib])" \\
      --run "python3 t2/report.py --eval t2/results"
"""
import argparse
import csv
import glob
import math
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry import (CUBE_HALF, DISCOVERY, MODES, geom_from_row,  # noqa: E402
                      half_extent, wilson)

# One colour per mode, used in every figure so a reader learns them once.
COLOR = {"nominal": "#4C72B0", "gap": "#C44E52", "farb": "#DD8452"}
LABEL = {"nominal": "nominal\n(reference)",
         "gap": "A: faces close\nface_gap < 20 mm",
         "farb": "B: cubeB far\ndist_B $\\geq$ 760 mm"}


def load_eval(d):
    """{mode: [rows_block1, rows_block2, rows_block3]}"""
    out = {}
    for f in sorted(glob.glob(f"{d}/mode_*_seed[0-9].csv")):
        m = re.match(r"mode_(.+)_seed(\d)\.csv$", os.path.basename(f))
        if m:
            out.setdefault(m.group(1), []).append(list(csv.DictReader(open(f))))
    return out


def stats(blocks):
    """Everything the table and the figure need, from a mode's 3 blocks."""
    rows = [r for b in blocks for r in b]
    n = len(rows)
    hit = lambda r, k: r[k] in ("1", 1, True)          # noqa: E731
    k = sum(hit(r, "success_once") for r in rows)
    g = sum(hit(r, "ever_grasped") for r in rows)
    p = sum(hit(r, "ever_placed") for r in rows)
    rates = [sum(hit(r, "success_once") for r in b) / len(b) for b in blocks]
    m = sum(rates) / len(rates)
    sd = (sum((x - m) ** 2 for x in rates) / (len(rates) - 1)) ** 0.5 \
        if len(rates) > 1 else float("nan")
    # `mean` (of the three block rates) is the DELIVERABLE point estimate - it
    # pairs with `sd`, and the pair is the error bar the 3-seed structure buys.
    # `pooled` (k/n over all 300) is what the Wilson interval is computed on, so
    # it is the estimate the stage-decomposition panel must plot: mixing a
    # mean-of-blocks point with a pooled-count interval puts the point outside
    # its own bar whenever the blocks are uneven.
    return dict(n=n, k=k, rates=rates, mean=m, sd=sd, pooled=k / n,
                ci=wilson(k, n),
                grasp=g / n, place=p / n, hold=k / max(p, 1),
                grasp_ci=wilson(g, n), place_ci=wilson(p, n), hold_ci=wilson(k, p))


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------


def table(ev):
    order = [t for t in ("nominal", "gap", "farb") if t in ev]
    print("\nPER-FAILURE-MODE SUCCESS RATE - 100 rollouts x 3 seeds")
    print(f"  {'mode':>9} {'per-block':>22} {'mean':>7} {'SD':>6} "
          f"{'pooled 95% CI':>16} {'n':>5}")
    for t in order:
        s = stats(ev[t])
        print(f"  {t:>9} {' '.join(f'{x:.3f}' for x in s['rates']):>22} "
              f"{s['mean']:7.3f} {s['sd']:6.3f} "
              f"[{s['ci'][0]:.3f}, {s['ci'][1]:.3f}] {s['n']:5d}")

    print("\nWHERE EACH MODE FAILS - the stage decomposition")
    print(f"  {'mode':>9} {'grasped':>9} {'placed':>9} {'held|placed':>13} "
          f"{'success':>9}")
    for t in order:
        s = stats(ev[t])
        print(f"  {t:>9} {s['grasp']:9.3f} {s['place']:9.3f} "
              f"{s['hold']:13.3f} {s['mean']:9.3f}")
    if "nominal" in ev and len(order) > 1:
        base = stats(ev["nominal"])
        print("\n  vs the nominal reference:")
        for t in order:
            if t == "nominal":
                continue
            s = stats(ev[t])
            print(f"  {t:>9} {s['grasp'] - base['grasp']:+9.3f} "
                  f"{s['place'] - base['place']:+9.3f} "
                  f"{s['hold'] - base['hold']:+13.3f} "
                  f"{s['mean'] - base['mean']:+9.3f}")
        print("\n  Mode A fails at PLACEMENT (place drops, hold|placed does not):"
              "\n  the descent onto cubeA fouls cubeB. Mode B fails at SETTLING"
              "\n  (grasp and place near baseline, hold|placed collapses): the arm"
              "\n  gets there and the stack does not stay. Different stages,"
              "\n  different mechanisms - which is what makes them two modes and"
              "\n  not two slices of one curve.")


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def fig_modes(ev, path):
    """The deliverable, and the mechanism, side by side."""
    order = [t for t in ("nominal", "gap", "farb") if t in ev]
    st = {t: stats(ev[t]) for t in order}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    x = np.arange(len(order))
    for i, t in enumerate(order):
        s = st[t]
        lo, hi = s["ci"]
        ax1.bar(i, s["mean"], 0.6, color=COLOR[t], alpha=.85, zorder=2)
        ax1.errorbar(i, s["mean"], yerr=[[s["mean"] - lo], [hi - s["mean"]]],
                     color="k", capsize=5, lw=1.4, zorder=3)
        # the three block rates, so the reader sees the spread the SD summarises
        ax1.scatter([i] * 3, s["rates"], color="k", s=18, zorder=4,
                    label="per-block rate" if i == 0 else None)
        ax1.text(i, hi + .03, f"{s['mean']:.3f}\n$\\pm${s['sd']:.3f}",
                 ha="center", fontsize=9)
    if "nominal" in st:
        ax1.axhline(st["nominal"]["mean"], color=COLOR["nominal"], ls="--",
                    lw=1, alpha=.7, zorder=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels([LABEL[t] for t in order], fontsize=9)
    ax1.set_ylabel("success_once")
    ax1.set_ylim(0, 1)
    ax1.set_title("Per-failure-mode success rate\n100 rollouts $\\times$ 3 seeds",
                  fontsize=11)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(axis="y", alpha=.3, zorder=0)

    # stage decomposition - the reason these are two modes
    stages = [("grasped", "grasp", "grasp_ci"), ("placed", "place", "place_ci"),
              ("held | placed", "hold", "hold_ci"), ("success", "pooled", "ci")]
    w = 0.8 / len(order)
    for i, t in enumerate(order):
        s = st[t]
        xs = np.arange(len(stages)) + (i - (len(order) - 1) / 2) * w
        vals = [s[k] for _, k, _ in stages]
        err = np.array([[s[k] - s[c][0] for _, k, c in stages],
                        [s[c][1] - s[k] for _, k, c in stages]])
        ax2.bar(xs, vals, w * .9, color=COLOR[t], alpha=.85,
                label=t, zorder=2)
        ax2.errorbar(xs, vals, yerr=err, fmt="none", ecolor="k", capsize=3,
                     lw=1, zorder=3)
    ax2.set_xticks(np.arange(len(stages)))
    ax2.set_xticklabels([s[0] for s in stages], fontsize=9)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("rate")
    ax2.set_title("Where each mode fails\nA loses PLACEMENT, B loses SETTLING",
                  fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=.3, zorder=0)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def binned(rows, key, edges):
    """(centres, rate, lo, hi, n) per bin, Wilson intervals."""
    c, r, lo, hi, ns = [], [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = [x for x in rows if a <= x[key] < b]
        if len(sel) < 8:
            continue
        k = sum(1 for x in sel if x["success_once"] == "1")
        l, h = wilson(k, len(sel))
        c.append((a + b) / 2 * 1000)
        r.append(k / len(sel)); lo.append(l); hi.append(h); ns.append(len(sel))
    return map(np.array, (c, r, lo, hi, ns))


def fig_axes(rows, path):
    """Success against the two axes the modes are defined on."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    overall = sum(1 for r in rows if r["success_once"] == "1") / len(rows)

    for ax, key, edges, xl, thr, tcol, note in (
        (axes[0], "face_gap", np.array([-.01, .01, .02, .035, .05, .08, .12, .20]),
         "face_gap  (mm)", 0.020, COLOR["gap"], "mode A"),
        (axes[1], "dist_B", np.array([.40, .50, .58, .64, .70, .76, .82, .90]),
         "dist_B, cubeB from the Panda base  (mm)", 0.76, COLOR["farb"], "mode B"),
    ):
        c, r, lo, hi, ns = binned(rows, key, edges)
        ax.errorbar(c, r, yerr=[r - lo, hi - r], fmt="o-", color="#333",
                    capsize=4, lw=1.4, ms=5, zorder=3)
        # above the upper CI bound, not above the point - otherwise the label
        # lands inside the error bar on the wide low-n bins
        for xi, hii, ni in zip(c, hi, ns):
            ax.text(xi, min(hii + .035, 0.98), f"n={ni}", ha="center", fontsize=7,
                    color="#666")
        ax.axhline(overall, color=COLOR["nominal"], ls="--", lw=1,
                   label=f"all episodes  {overall:.3f}", zorder=1)
        ax.axvline(thr * 1000, color=tcol, lw=1.6, alpha=.8,
                   label=f"{note} threshold", zorder=2)
        ax.set_xlabel(xl)
        ax.set_ylim(0, 1)
        ax.grid(alpha=.3, zorder=0)
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("success_once")
    axes[0].set_title("Clearance between the cubes' faces", fontsize=11)
    axes[1].set_title("How far cubeB is from the arm", fontsize=11)
    fig.suptitle(f"A failure mode is a REGION of the initial-state distribution "
                 f"({len(rows)} held-out episodes, Wilson 95%)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_face_gap(path):
    """What face_gap is. No data - this is a definition, drawn."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, (ta, tb, title) in zip(axes, [
            (0.0, 0.0, "both cubes square to the bearing"),
            (math.radians(45), math.radians(45), "both turned 45$\\degree$")]):
        sep = 0.060
        A, B = (0.0, 0.0), (sep, 0.0)
        psi = 0.0
        for (cx, cy), th, col, nm in ((A, ta, "#C44E52", "cubeA"),
                                      (B, tb, "#4C72B0", "cubeB")):
            corners = [(cx + CUBE_HALF * math.cos(th + k) * math.sqrt(2),
                        cy + CUBE_HALF * math.sin(th + k) * math.sqrt(2))
                       for k in np.arange(4) * math.pi / 2 + math.pi / 4]
            ax.add_patch(plt.Polygon(corners, closed=True, facecolor=col,
                                     alpha=.35, edgecolor=col, lw=2))
            ax.plot(*zip(*[(cx, cy)]), "o", color=col, ms=4)
            ax.text(cx, cy - 0.046, nm, ha="center", color=col, fontsize=10)

        eA = half_extent(psi - ta)
        eB = half_extent(psi + math.pi - tb)
        gap = sep - eA - eB
        # separation, then each half-extent, then what is left
        ax.annotate("", (A[0], .038), (B[0], .038),
                    arrowprops=dict(arrowstyle="<->", color="#666"))
        ax.text(sep / 2, .043, f"separation = {sep * 1000:.0f} mm",
                ha="center", fontsize=9, color="#666")
        for x0, x1, col in ((A[0], A[0] + eA, "#C44E52"),
                            (B[0] - eB, B[0], "#4C72B0")):
            ax.annotate("", (x0, -.030), (x1, -.030),
                        arrowprops=dict(arrowstyle="<->", color=col))
        ax.text(A[0] + eA / 2, -.040, f"{eA * 1000:.1f}", ha="center",
                fontsize=8, color="#C44E52")
        ax.text(B[0] - eB / 2, -.040, f"{eB * 1000:.1f}", ha="center",
                fontsize=8, color="#4C72B0")
        # Below ~6 mm the double-headed arrow is shorter than its own arrowheads
        # and renders as a bowtie, which reads as a glitch. The label carries the
        # number; the picture only has to show that the faces nearly touch.
        if gap > 0.006:
            ax.annotate("", (A[0] + eA, .012), (B[0] - eB, .012),
                        arrowprops=dict(arrowstyle="<->", color="k", lw=2))
        ax.text(sep / 2, .018, f"face_gap = {gap * 1000:.1f} mm", ha="center",
                fontsize=10, fontweight="bold")
        ax.set_xlim(-.055, .115)
        ax.set_ylim(-.055, .058)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=10)

    fig.suptitle("face_gap = separation $-$ extent$_A$(bearing) $-$ extent$_B$"
                 "(bearing)\nSame centres, same 60 mm separation. Yaw is the "
                 "difference, and the gripper cannot rotate to compensate.",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval", default=None, help="dir with mode_*_seed*.csv")
    p.add_argument("--discovery", default=None, help="the nominal-pass CSV")
    p.add_argument("--figdir", default="figures")
    a = p.parse_args()
    os.makedirs(a.figdir, exist_ok=True)
    wrote = []

    fig_face_gap(f"{a.figdir}/t2_face_gap.png")
    wrote.append("t2_face_gap.png")

    if a.eval:
        ev = load_eval(a.eval)
        if ev:
            table(ev)
            fig_modes(ev, f"{a.figdir}/t2_modes.png")
            wrote.append("t2_modes.png")
        else:
            print(f"  (no mode_*_seed*.csv under {a.eval} - skipping the "
                  f"deliverable table)")

    if a.discovery and os.path.exists(a.discovery):
        rows = list(csv.DictReader(open(a.discovery)))
        for r in rows:
            r.update(geom_from_row(r))
        fig_axes(rows, f"{a.figdir}/t2_axes.png")
        wrote.append("t2_axes.png")
        print(f"\nDISCOVERY PASS - {len(rows)} held-out episodes, "
              f"the evidence the modes are regions")
        print(f"  {'region':>9} {'n':>6} {'success':>9} {'95% CI':>16} "
              f"{'grasp':>7} {'place':>7} {'hold|pl':>8}")
        for tag in ("nominal", "gap", "farb"):
            sel = [r for r in rows if MODES[tag](r)]
            n = len(sel)
            k = sum(1 for r in sel if r["success_once"] == "1")
            g = sum(1 for r in sel if r["ever_is_cubeA_grasped"] == "1")
            pl = sum(1 for r in sel if r["ever_is_cubeA_on_cubeB"] == "1")
            lo, hi = wilson(k, n)
            print(f"  {tag:>9} {n:6d} {k / n:9.3f} [{lo:.3f}, {hi:.3f}] "
                  f"{g / n:7.3f} {pl / n:7.3f} {k / max(pl, 1):8.3f}")
        both = sum(1 for r in rows if MODES["gap"](r) and MODES["farb"](r))
        print(f"\n  episodes in BOTH regions: {both}   "
              f"(must be 0 - the modes are disjoint by construction)")

    print(f"\nwrote {a.figdir}/: " + " ".join(wrote))


if __name__ == "__main__":
    main()
