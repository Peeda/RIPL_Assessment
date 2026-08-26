#!/usr/bin/env python
"""The T-IV figures and tables: training curves, and the before/after.

Needs numpy and matplotlib, so it sits ABOVE the stdlib layer - see CLAUDE.md's
nix-shell note. Nothing here decides anything; verify_t4.py does that.

    nix-shell -p "python3.withPackages(ps: [ps.numpy ps.matplotlib])" --run \\
      "python3 t4/report.py --runs t4/runs --before t2/results \\
         --after gap=t4/results/after_gap farb=t4/results/after_farb"

Two figures, both written to figures/ and committed:

  t4_training.png   reward, success, the losses, the alpha ramp against the
                    measured |delta|, and VRAM/throughput. The assignment asks
                    for reward/loss/VRAM curves and wall-clock by name.
  t4_before_after.png  per mode, the three blocks before and after, with Wilson
                    intervals on the pooled rate.
"""
import argparse
import csv
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "t2"))
from geometry import wilson  # noqa: E402

# One colour per mode, used in both figures so a reader can carry them across.
CMAP = {"nominal": "#4c72b0", "gap": "#c44e52", "farb": "#dd8452"}


def read_log(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    out = {}
    for k in rows[0]:
        v = []
        for r in rows:
            try:
                v.append(float(r[k]) if r[k] != "" else np.nan)
            except ValueError:
                v.append(np.nan)
        out[k] = np.asarray(v)
    return out


def read_pass(d):
    """{mode: {block: rows}} plus the manifests."""
    out, man = {}, {}
    for p in sorted(glob.glob(os.path.join(d, "mode_*_seed[0-9].csv"))):
        stem = os.path.basename(p)[:-4]
        mode = stem[len("mode_"):stem.rindex("_seed")]
        block = int(stem[stem.rindex("_seed") + 5:])
        with open(p) as f:
            out.setdefault(mode, {})[block] = list(csv.DictReader(f))
        mp = os.path.join(d, stem + "_manifest.json")
        if os.path.exists(mp):
            man[(mode, block)] = json.load(open(mp))
    return out, man


def rates(blocks, key="success_once"):
    per = [np.mean([int(r[key]) for r in blocks[b]]) for b in sorted(blocks)]
    k = sum(int(r[key]) for b in blocks for r in blocks[b])
    n = sum(len(blocks[b]) for b in blocks)
    return np.asarray(per), k, n


# ---------------------------------------------------------------------------
# figure 1 - training
# ---------------------------------------------------------------------------


def fig_training(runs_dir, out):
    logs = {}
    for p in sorted(glob.glob(os.path.join(runs_dir, "*", "*_train.csv"))):
        mode = os.path.basename(os.path.dirname(p))
        logs.setdefault(mode, []).append((os.path.basename(p)[:-10], read_log(p)))
    if not logs:
        print(f"  (no *_train.csv under {runs_dir}; skipping the training figure)")
        return None

    panels = [
        ("train/success_once", "success_once (training distribution)", None),
        ("charts/reward_mean", "mean chunk reward  [0, 1]", None),
        ("losses/policy_loss", "policy loss", None),
        ("losses/value_loss", "value loss", "log"),
        ("losses/approx_kl", "approx KL (early-stop at 0.1)", "log"),
        ("losses/entropy", "policy entropy", None),
        ("charts/delta_norm_mm", "applied |delta| (mm/step)", None),
        ("sys/vram_max_gb", "peak VRAM (GB)", None),
        ("charts/SPS", "throughput (env steps/s)", None),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(15, 10.5))
    for ax, (key, title, scale) in zip(axes.ravel(), panels):
        drew = False
        for mode, runs in sorted(logs.items()):
            for name, L in runs:
                if key not in L or np.all(np.isnan(L[key])):
                    continue
                ax.plot(L["global_step"] / 1e6, L[key], lw=1.2,
                        color=CMAP.get(mode, "#555555"), alpha=0.85,
                        label=name if not drew or True else None)
                drew = True
        if key == "charts/delta_norm_mm":
            # the bound the residual is working against, so saturation is
            # visible rather than inferred
            for mode, runs in sorted(logs.items()):
                for name, L in runs:
                    if "charts/alpha_mm" in L:
                        ax.plot(L["global_step"] / 1e6, L["charts/alpha_mm"],
                                lw=1.0, ls="--", color="#333333",
                                label="alpha bound" if not drew else None)
                        break
                break
        if key == "losses/approx_kl":
            ax.axhline(0.1, color="#333333", ls="--", lw=1.0)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("env steps (millions)", fontsize=8)
        if scale:
            ax.set_yscale(scale)
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=8)
        if not drew:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="#999999")
    handles = [plt.Line2D([], [], color=CMAP.get(m, "#555"), lw=2, label=m)
               for m in sorted(logs)]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=10)
    fig.suptitle("T-IV: PPO residual training, per mode, 3 seeds each", y=0.995)
    fig.tight_layout(rect=(0, 0.035, 1, 0.98))
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  wrote {out}")
    return out


# ---------------------------------------------------------------------------
# figure 2 - before / after
# ---------------------------------------------------------------------------


def fig_before_after(before, afters, out):
    modes = ["nominal", "gap", "farb"]
    modes = [m for m in modes if m in before]
    fig, axes = plt.subplots(1, len(afters), figsize=(6.2 * len(afters), 4.6),
                             squeeze=False)
    for ax, (target, after) in zip(axes[0], sorted(afters.items())):
        xs = np.arange(len(modes))
        for off, src, lab, hatch in ((-0.19, before, "base", None),
                                     (0.19, after, "residual", "//")):
            hs, los, his, pts = [], [], [], []
            for m in modes:
                if m not in src:
                    hs.append(np.nan); los.append(0); his.append(0); pts.append([])
                    continue
                per, k, n = rates(src[m])
                lo, hi = wilson(k, n)
                hs.append(k / n); los.append(k / n - lo); his.append(hi - k / n)
                pts.append(per)
            ax.bar(xs + off, hs, 0.34, yerr=[los, his], capsize=4,
                   color=[CMAP.get(m, "#999") for m in modes],
                   alpha=0.55 if lab == "base" else 1.0, hatch=hatch,
                   edgecolor="white", label=lab)
            for x, p in zip(xs + off, pts):
                if len(p):
                    ax.plot([x] * len(p), p, "k.", ms=5, alpha=0.75, zorder=3)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{m}\n(target)" if m == target else m for m in modes])
        ax.set_ylim(0, 1)
        ax.set_ylabel("success_once")
        ax.axhline(0, color="k", lw=0.8)
        ax.grid(axis="y", alpha=0.25, lw=0.5)
        ax.set_title(f"residual trained on '{target}'")
        ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("T-IV: 100 rollouts x 3 seeds, held-out seeds, physx_cpu. "
                 "Dots are the three blocks; bars are the pooled rate with a "
                 "Wilson 95% interval.", fontsize=9, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")
    return out


# ---------------------------------------------------------------------------


def table(before, afters):
    print("\n" + "=" * 92)
    print(f"{'target':>8} {'mode':>9}  {'before':>24}   {'after':>24}   {'delta':>15}")
    for target, after in sorted(afters.items()):
        for m in ("nominal", "gap", "farb"):
            if m not in before or m not in after:
                continue
            pb, kb, nb = rates(before[m])
            pa, ka, na = rates(after[m])
            lb, hb = wilson(kb, nb)
            la, ha = wilson(ka, na)
            d = ka / na - kb / nb
            mark = "  <- target" if m == target else ""
            print(f"{target:>8} {m:>9}  {kb/nb:.3f} [{lb:.3f},{hb:.3f}] "
                  f"sd {pb.std():.3f}   {ka/na:.3f} [{la:.3f},{ha:.3f}] "
                  f"sd {pa.std():.3f}   {d:+.3f}{mark}")
    print("\n  The intervals are Wilson at the pooled n; the sd is over the "
          "three blocks and\n  is the honest error bar, because the blocks are "
          "the independent replicates.")


def wallclock(runs_dir):
    rows = []
    for p in sorted(glob.glob(os.path.join(runs_dir, "*", "*_manifest.json"))):
        m = json.load(open(p))
        if "wall_seconds" not in m:
            continue
        rows.append((m.get("mode", "?"), m.get("seed", "?"), m.get("gpu", "?"),
                     m.get("env_steps", 0), m["wall_seconds"],
                     m.get("vram_max_gb", 0), m.get("alpha", 0),
                     m.get("res_horizon", 0)))
    if not rows:
        return
    print("\n" + "=" * 92)
    print(f"{'mode':>8} {'seed':>5} {'alpha':>6} {'res_h':>6} {'env steps':>12} "
          f"{'wall':>9} {'VRAM':>7}  gpu")
    for mode, seed, gpu, steps, wall, vram, al, rh in rows:
        print(f"{mode:>8} {seed:>5} {al:>6.3f} {rh:>6} {steps:>12,} "
              f"{wall/60:>7.1f}m {vram:>6.1f}G  {gpu}")


def main():
    ap = argparse.ArgumentParser()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--runs", default=os.path.join(root, "t4", "runs"))
    ap.add_argument("--before", default=os.path.join(root, "t2", "results"))
    ap.add_argument("--after", nargs="*", default=[],
                    help="target=dir, e.g. gap=t4/results/after_gap")
    ap.add_argument("--figdir", default=os.path.join(root, "figures"))
    a = ap.parse_args()

    os.makedirs(a.figdir, exist_ok=True)
    fig_training(a.runs, os.path.join(a.figdir, "t4_training.png"))
    wallclock(a.runs)

    if a.after:
        before, _ = read_pass(a.before)
        afters = {}
        for spec in a.after:
            tgt, d = spec.split("=", 1)
            afters[tgt] = read_pass(d)[0]
        table(before, afters)
        fig_before_after(before, afters,
                         os.path.join(a.figdir, "t4_before_after.png"))
    else:
        print("  (no --after given; skipping the before/after figure)")


if __name__ == "__main__":
    main()
