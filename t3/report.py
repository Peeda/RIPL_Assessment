#!/usr/bin/env python
"""T-III's figures. Needs numpy and matplotlib.

    python3 t3/report.py --out $T3_OUT --mode gap --figdir figures \
                         --index t2/results/seeds.csv

Two figures, one per deliverable:

  t3_sampler_<mode>.png    deliverable (b): the biased distribution against the
                           environment's own, on the axis that defines the
                           region. The picture that shows the sampler did
                           something.
  t3_alignment_<mode>.png  deliverable (a): cumulative generated reward by the
                           stage each episode reached, generated beside stock.
                           If the reward is useful, the boxes step upward.

On the laptop:
    nix-shell -p "python3.withPackages(ps: [ps.numpy ps.matplotlib])" \
      --run "python3 t3/report.py --out t3/results --mode gap"
"""
import argparse
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(os.path.dirname(HERE), "t2"))
from geometry import geom_from_row  # noqa: E402

GEN, STOCK, NOM = "#1f4e79", "#c0703a", "#9aa5ad"


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fig_sampler(out, mode, figdir, index):
    sp = os.path.join(out, f"sampler_{mode}.csv")
    if not os.path.exists(sp):
        return
    s = rows(sp)
    gap = np.array([float(r["face_gap"]) for r in s]) * 1000
    dB = np.array([float(r["dist_B"]) for r in s]) * 1000

    nom_gap = nom_dB = None
    if index and os.path.exists(index):
        n = [geom_from_row(r) for r in rows(index)]
        nom_gap = np.array([g["face_gap"] for g in n]) * 1000
        nom_dB = np.array([g["dist_B"] for g in n]) * 1000

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for a, vals, nom, lab, thr in (
            (ax[0], gap, nom_gap, "face gap (mm)", 20 if mode == "gap" else None),
            (ax[1], dB, nom_dB, "cubeB distance from base (mm)",
             760 if mode == "farb" else None)):
        if nom is not None:
            a.hist(nom, bins=50, density=True, color=NOM, alpha=0.85,
                   label="environment's own sampler")
        a.hist(vals, bins=50, density=True, color=GEN, alpha=0.75,
               label="LLM-generated sampler")
        if thr is not None:
            a.axvline(thr, color="k", ls="--", lw=1,
                      label=f"region boundary ({thr} mm)")
        a.set_xlabel(lab)
        a.set_ylabel("density")
        a.legend(fontsize=7, frameon=False)
    j = os.path.join(out, f"sampler_{mode}.json")
    if os.path.exists(j):
        m = json.load(open(j))
        fig.suptitle(f"mode '{mode}': {m['hit_rate']:.0%} of draws in region "
                     f"against a nominal {m['base_rate']:.1%} "
                     f"({m['enrichment']:.0f}x)", fontsize=9)
    fig.tight_layout()
    p = os.path.join(figdir, f"t3_sampler_{mode}.png")
    fig.savefig(p, dpi=150)
    print(f"  wrote {p}")


def _stage(r):
    if r["success_once"] == "1":
        return 2
    if r["ever_placed"] == "1":
        return 1
    if r["ever_grasped"] == "1":
        return 0
    return -1


def fig_alignment(out, mode, figdir):
    ap = os.path.join(out, f"align_{mode}.csv")
    if not os.path.exists(ap):
        return
    arms = [("generated", ap, GEN)]
    sp = os.path.join(out, "align_stock.csv")
    if os.path.exists(sp):
        arms.append(("ManiSkill stock", sp, STOCK))

    labels = ["grasped,\nnot placed", "placed,\nnot success", "success"]
    fig, ax = plt.subplots(1, len(arms), figsize=(4.4 * len(arms), 3.6),
                           squeeze=False)
    for k, (name, path, colour) in enumerate(arms):
        a = ax[0][k]
        pol = [r for r in rows(path) if r["arm"] == "policy"]
        # Normalised by REWARD_MAX * ep_len so the two panels share an axis
        # whatever scale the model chose. The GATE stays scale-free (AUCs and
        # ratios); only the picture needs a common unit.
        mj = os.path.join(out, f"align_{'stock' if k else mode}_manifest.json")
        rmax = json.load(open(mj))["reward_max"] if os.path.exists(mj) else 1.0
        data = [[float(r["ep_return"]) / (rmax * max(int(r["ep_len"]), 1))
                 for r in pol if _stage(r) == s] for s in (0, 1, 2)]
        keep = [(d, l) for d, l in zip(data, labels) if d]
        # matplotlib renamed `labels` to `tick_labels` in 3.9; the laptop has
        # 3.11 and the pod has whatever pip resolved. Try the new name first.
        vals, labs = [d for d, _ in keep], [l for _, l in keep]
        try:
            bp = a.boxplot(vals, tick_labels=labs, patch_artist=True, widths=0.55)
        except TypeError:
            bp = a.boxplot(vals, labels=labs, patch_artist=True, widths=0.55)
        for box in bp["boxes"]:
            box.set(facecolor=colour, alpha=0.6)
        for med in bp["medians"]:
            med.set(color="k", lw=1.4)
        a.set_title(name, fontsize=10)
        a.set_ylabel("mean reward per step (fraction of max)")
        a.tick_params(labelsize=8)
    fig.suptitle(f"mode '{mode}': does cumulative reward rise with the stage "
                 f"the episode reached?", fontsize=9)
    fig.tight_layout()
    p = os.path.join(figdir, f"t3_alignment_{mode}.png")
    fig.savefig(p, dpi=150)
    print(f"  wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--figdir", default="figures")
    ap.add_argument("--index")
    a = ap.parse_args()
    os.makedirs(a.figdir, exist_ok=True)
    fig_sampler(a.out, a.mode, a.figdir, a.index)
    fig_alignment(a.out, a.mode, a.figdir)
    print(f"\n  Look at them. CLAUDE.md's rule: a figure that exists is not a "
          f"figure that is right.")


if __name__ == "__main__":
    main()
