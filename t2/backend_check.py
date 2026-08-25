#!/usr/bin/env python
"""Does physx_cuda reproduce physx_cpu for this policy?

T-IV has to train its PPO residual on physx_cuda - physx_cpu vectorises by
subprocess, and at the measured ~400k env-steps/hour a 10M-step run is ~25 h per
seed. But T-I and every T-II number were measured on physx_cpu, and the
seed-addressed harness CANNOT be ported:

  * StackCube's _initialize_episode draws cube poses with torch.rand under
    `with torch.device(self.device)` (stack_cube.py:78-100), so the CUDA
    generator produces different states than the CPU one for the same seed;
  * reset() seeds the whole batch from self._episode_seed[0]
    (sapien_env.py:950-953) while _initialize_episode draws (b, 2) at once, so
    on GPU a seed does not even address one episode - it addresses a batch.

The second is a live trap: mine_rollouts.py asserts the episode seed came back
as requested, and on GPU that assert PASSES AND LIES, because _episode_seed is
recorded per-env while only element 0 drives the sampling.

So the plan is train on GPU, evaluate on CPU: the policy transfers, the states
do not. This script is what licenses that split. It sidesteps seeds entirely by
replaying the CPU pass's EXACT initial states on the GPU through
options={"reset_to_env_states": ...}, which batches natively there. The CPU arm
is the committed nominal.csv - no CPU re-run needed.

The verdict is NOT "agreement == 1.0". DDPM action sampling is stochastic, so
even CPU-vs-CPU on an identical initial state disagrees; the region passes
carry repeats=2 and measure that floor at 0.74 / 0.67. The question is whether
GPU-vs-CPU lands at the floor (backends equivalent) or below it (they are not),
and whether the disagreement is symmetric (noise) or directional (a real shift),
which is what McNemar tests.

  BACKEND=physx_cuda python t2/backend_check.py CKPT \\
      --states results/nominal_states.npz --cpu-csv results/nominal.csv
"""
import argparse
import csv
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t2_common import (build_agent, geom_from_row,  # noqa: E402
                       manifest, state_dict_from_flat, to_device, wilson)

MAX_EP_STEPS = 200


def np1(x):
    """Any per-env info value -> a flat numpy array.

    physx_cpu hands back numpy via CPUGymWrapper; physx_cuda hands back cuda
    tensors from ManiSkillVectorEnv. np.asarray on a cuda tensor raises, so
    everything crossing out of info goes through here.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().reshape(-1)
    return np.asarray(x).reshape(-1)


def mcnemar(b, c):
    """Two-sided exact-ish McNemar on the discordant counts.

    b = CPU succeeded and GPU failed, c = the reverse. Only the discordant
    pairs carry information about a directional shift; the concordant ones are
    the same answer twice and say nothing. Normal approximation with continuity
    correction, which is fine at the counts this run produces.
    """
    n = b + c
    if n == 0:
        return 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / n
    p = math.erfc(math.sqrt(chi2 / 2))
    return chi2, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--states", required=True, help="*_states.npz from mine_rollouts.py")
    ap.add_argument("--cpu-csv", required=True, help="the matching *.csv")
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="0 = every episode")
    ap.add_argument("--state", action="store_true")
    ap.add_argument("--out", default="backend_check")
    a = ap.parse_args()

    backend = os.environ.get("BACKEND", "")
    if backend != "physx_cuda":
        print(f"!! BACKEND={backend or '(unset)'}; this check only means something "
              f"with BACKEND=physx_cuda, since the CPU arm is the committed CSV.")
        sys.exit(1)

    cpu_rows = list(csv.DictReader(open(a.cpu_csv)))
    z = np.load(a.states, allow_pickle=True)
    init, names = z["init"], z["names"]
    if len(init) != len(cpu_rows):
        sys.exit(f"!! {a.states} has {len(init)} states but {a.cpu_csv} has "
                 f"{len(cpu_rows)} rows; they must be row-aligned.")
    n = a.limit or len(cpu_rows)
    cpu_rows, init = cpu_rows[:n], init[:n]
    print(f"replaying {n} CPU episodes on physx_cuda, width {a.num_envs}")

    agent, envs, args, device = build_agent(a.ckpt, a.state, a.num_envs,
                                            max_episode_steps=MAX_EP_STEPS)
    meta = manifest(ckpt=os.path.abspath(a.ckpt), backend=backend,
                    source_csv=os.path.abspath(a.cpu_csv), episodes=n,
                    num_envs=a.num_envs, max_episode_steps=MAX_EP_STEPS)
    for k, v in meta.items():
        print(f"  {k:16s} {v}")
    print("", flush=True)

    gpu = np.zeros(n, dtype=int)
    for start in range(0, n, a.num_envs):
        batch = list(range(start, min(start + a.num_envs, n)))
        # Pad by repeating the last state rather than rebuilding the vec env at
        # a narrower width; padding rows are dropped below.
        idx = batch + [batch[-1]] * (a.num_envs - len(batch))
        sd = state_dict_from_flat(names, init[idx])
        obs, info = envs.reset(options={"reset_to_env_states": {"env_states": sd}})

        # There is no seed here, so mine_rollouts.py's seed assert has no
        # analogue. This is the honest replacement: prove the injection took by
        # reading the poses back and matching them against the CPU row.
        a0 = np1(info["cubeA_pose"]).reshape(a.num_envs, -1)
        for j, i in enumerate(batch):
            want = (float(cpu_rows[i]["cubeA_x"]), float(cpu_rows[i]["cubeA_y"]))
            got = (float(a0[j][0]), float(a0[j][1]))
            if abs(got[0] - want[0]) > 1e-5 or abs(got[1] - want[1]) > 1e-5:
                sys.exit(f"\n  env {j} reset to cubeA {got}, expected {want}.\n"
                         f"  The state injection did not take, so the two arms "
                         f"would describe different episodes.\n")

        ever = np.zeros(a.num_envs, bool)
        steps, done = 0, False
        while not done and steps < MAX_EP_STEPS:
            with torch.no_grad():
                chunk = agent.get_action(to_device(obs, device)).cpu().numpy()
            for i in range(chunk.shape[1]):
                obs, rew, term, trunc, info = envs.step(chunk[:, i])
                steps += 1
                if "success" in info:
                    ever |= np1(info["success"]).astype(bool)
                if bool(np1(trunc).any()) or bool(np1(term).any()):
                    done = True
                    break
        # success_once computed from our own accumulator, not the wrapper's key,
        # so both arms use an identically-defined metric across two different
        # wrapper stacks (CPUGymWrapper vs ManiSkillVectorEnv).
        for j, i in enumerate(batch):
            gpu[i] = int(ever[j])
        print(f"  {min(start + a.num_envs, n):>5}/{n}  "
              f"gpu success_once so far {gpu[:min(start + a.num_envs, n)].mean():.3f}",
              flush=True)

    envs.close()

    # ---- compare ----------------------------------------------------------
    cpu = np.array([int(r.get("ever_success") or r["success_once"]) for r in cpu_rows])
    both = int(((cpu == 1) & (gpu == 1)).sum())
    neither = int(((cpu == 0) & (gpu == 0)).sum())
    b = int(((cpu == 1) & (gpu == 0)).sum())     # CPU yes, GPU no
    c = int(((cpu == 0) & (gpu == 1)).sum())
    agree = (both + neither) / n
    chi2, p = mcnemar(b, c)

    print("\n" + "=" * 68)
    print("MARGINAL")
    for lab, v in (("physx_cpu", cpu), ("physx_cuda", gpu)):
        lo, hi = wilson(int(v.sum()), n)
        print(f"  {lab:11s} success_once {v.mean():.3f}  [{lo:.3f}, {hi:.3f}]  n={n}")
    print(f"  difference  {gpu.mean() - cpu.mean():+.3f}")

    print("\nPAIRED (identical initial states)")
    print(f"  both succeed {both:5d}   both fail {neither:5d}")
    print(f"  cpu only     {b:5d}   gpu only  {c:5d}")
    print(f"  agreement    {agree:.3f}")
    print(f"  McNemar      chi2={chi2:.2f}  p={p:.3f}   "
          f"({'no directional shift' if p > 0.05 else 'DIRECTIONAL SHIFT'})")
    print("\n  Compare agreement against the same-backend floor, not against")
    print("  1.0 - DDPM sampling is stochastic, so identical states disagree")
    print("  even CPU-vs-CPU. The region passes measure that floor at ~0.74.")

    # ---- conditional: does the failure REGION move? -----------------------
    print("\nCONDITIONAL - the marginal agreeing while the region moves is the")
    print("bad outcome, and only this catches it.")
    for key, edges in (("face_gap", [-1, 0.02, 0.04, 0.07, 0.12, 9]),
                       ("dist_max", [0, 0.60, 0.68, 0.76, 9])):
        print(f"\n  {key}")
        print(f"    {'bin':>16}  {'n':>5}  {'cpu':>6}  {'gpu':>6}  {'diff':>6}")
        feats = [geom_from_row(r)[key] for r in cpu_rows]
        for lo, hi in zip(edges, edges[1:]):
            m = np.array([lo <= f < hi for f in feats])
            if m.sum() < 5:
                continue
            print(f"    {lo:6.3f}-{hi:6.3f}  {int(m.sum()):5d}  "
                  f"{cpu[m].mean():6.3f}  {gpu[m].mean():6.3f}  "
                  f"{gpu[m].mean() - cpu[m].mean():+6.3f}")

    with open(a.out + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "seed", "cpu_success_once", "gpu_success_once"])
        for i, r in enumerate(cpu_rows):
            w.writerow([i, r.get("seed", ""), cpu[i], gpu[i]])
    meta.update(cpu_success_once=float(cpu.mean()), gpu_success_once=float(gpu.mean()),
                agreement=agree, mcnemar_chi2=chi2, mcnemar_p=p,
                cpu_only=b, gpu_only=c)
    with open(a.out + "_manifest.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nwrote {a.out}.csv and {a.out}_manifest.json")


if __name__ == "__main__":
    main()
