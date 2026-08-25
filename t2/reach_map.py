#!/usr/bin/env python
"""Where can the Panda actually put its gripper, with the orientation it is stuck with?

The far-reach failure region sits inside the Panda's spec reach - over 8,000
indexed seeds dist_max tops out at 839 mm against a 855 mm spec, and the
sampler's theoretical corner is 869 mm. So "the cube is out of reach" is not
true in the free-orientation sense, and that is the trivial explanation ruled
out.

But spec reach assumes the arm may choose any wrist orientation, and this task
does not get to. Under pd_ee_delta_pos the action is padded with three zeros
and compute_target_pose keeps the current rotation (pd_ee_pose.py:86-99), so
the gripper's orientation is FROZEN at whatever it was at reset for the entire
episode. The reachable set for a fixed top-down orientation is a good deal
smaller than the spec sphere, and that is the set this measures.

Policy-free and GPU-free, the same shape as seed_index.py: it asks the
controller's own Kinematics, not a formula. On physx_cpu compute_ik goes
through Pinocchio and returns None when it fails to converge, which is a real
verdict rather than a best effort - but it is still a LOCAL solver seeded from
q0, so a solution is confirmed by forward kinematics and a position residual
rather than trusted.

  python t2/reach_map.py --out t2/results/reach_map.csv
"""
import argparse
import csv
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t2_common import PANDA_BASE_XY  # noqa: E402

RESIDUAL_TOL = 0.005          # 5 mm: a quarter of a cube half-width


def build_env(max_episode_steps=200):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    env_id = os.environ.get("ENV_ID", "StackCube-v1")
    ctrl = os.environ.get("CTRL", "pd_ee_delta_pos")
    env = gym.make(env_id, obs_mode="state", control_mode=ctrl,
                   sim_backend="physx_cpu", reconfiguration_freq=1,
                   max_episode_steps=max_episode_steps)
    env.reset(seed=0)
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reach_map.csv")
    ap.add_argument("--step", type=float, default=0.01, help="grid pitch, m")
    ap.add_argument("--z", type=float, default=0.02,
                    help="TCP height, m. 0.02 is a cube's centre - where the "
                         "gripper has to be to close on it.")
    ap.add_argument("--figdir", default="figures")
    a = ap.parse_args()

    env = build_env()
    u = env.unwrapped
    agent = u.agent
    # pd_ee_delta_pos is a CombinedController: an arm controller plus a gripper
    # controller, exposed as .controllers (a dict of uid -> controller), not as
    # a dict itself. Find the one that actually carries a Kinematics rather
    # than guessing by uid - the naming is not stable across robots.
    ctrl = agent.controller
    kin = getattr(ctrl, "kinematics", None)
    if kin is None:
        subs = getattr(ctrl, "controllers", None) or (ctrl if isinstance(ctrl, dict) else {})
        for uid, c in subs.items():
            if hasattr(c, "kinematics"):
                ctrl, kin = c, c.kinematics
                print(f"  arm controller  '{uid}'  ({type(c).__name__})")
                break
    if kin is None:
        sys.exit("!! no sub-controller exposes .kinematics; cannot solve IK.\n"
                 f"   controller is {type(agent.controller).__name__} with "
                 f"{list(getattr(agent.controller, 'controllers', {}))}")

    q_home = agent.robot.get_qpos().clone()
    tcp = agent.tcp.pose
    quat = tcp.q.clone()                       # the orientation it is stuck with
    w, x, y, z = [float(v) for v in np.asarray(tcp.q.cpu()).reshape(-1)[:4]]
    gripper_yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    print(f"  home TCP    p={np.asarray(tcp.p.cpu()).reshape(-1).round(4).tolist()}")
    print(f"  frozen quat {np.asarray(tcp.q.cpu()).reshape(-1).round(4).tolist()}")
    print(f"  gripper yaw {math.degrees(gripper_yaw):+.1f} deg  <- fixed for the")
    print(f"              whole episode; the policy cannot rotate to square up")
    print(f"              to a cube, which is why face_gap and not separation.")

    from mani_skill.utils.structs.pose import Pose
    idx = np.asarray(ctrl.active_joint_indices).reshape(-1)

    # Jacobian for manipulability. The CPU path builds a Pinocchio model and no
    # pytorch_kinematics chain (kinematics.py _setup_cpu), so this is best
    # effort - a missing manipulability column is not worth failing the run for.
    pmodel = getattr(kin, "pmodel", None)

    xs = np.arange(-0.24, 0.2401, a.step)
    ys = np.arange(-0.34, 0.3401, a.step)
    rows, n_ok = [], 0
    for gx in xs:
        for gy in ys:
            target = Pose.create_from_pq(
                p=torch.tensor([[float(gx), float(gy), a.z]], dtype=torch.float32),
                q=quat)
            try:
                q = kin.compute_ik(target, q0=q_home)
            except Exception:
                q = None
            residual, manip = float("nan"), float("nan")
            if q is not None:
                qf = q_home.clone()
                qf[:, idx] = q.to(qf.dtype)
                agent.robot.set_qpos(qf)
                p = np.asarray(agent.tcp.pose.p.cpu()).reshape(-1)
                residual = float(np.linalg.norm(p - np.array([gx, gy, a.z])))
                if pmodel is not None:
                    try:
                        pmodel.compute_forward_kinematics(
                            qf[:, kin.pmodel_active_joint_indices].cpu().numpy()[0])
                        J = np.asarray(pmodel.compute_single_link_local_jacobian(
                            qf[:, kin.pmodel_active_joint_indices].cpu().numpy()[0],
                            kin.end_link_idx))[:3]
                        manip = float(math.sqrt(max(np.linalg.det(J @ J.T), 0.0)))
                    except Exception:
                        pmodel = None          # say it once, not per grid cell
                agent.robot.set_qpos(q_home)
            ok = int(q is not None and residual == residual and residual <= RESIDUAL_TOL)
            n_ok += ok
            rows.append(dict(x=float(gx), y=float(gy), z=a.z,
                             dist=math.dist((gx, gy), PANDA_BASE_XY),
                             ik_ok=int(q is not None), residual=residual,
                             reachable=ok, manipulability=manip))
    env.close()

    with open(a.out, "w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w_.writeheader()
        w_.writerows(rows)
    print(f"\n  {n_ok}/{len(rows)} grid points reachable within "
          f"{RESIDUAL_TOL*1000:.0f} mm at z={a.z}")

    # The number the far-mode argument actually turns on: where does the
    # reachable set end, along the axis T-II measures failures on?
    print(f"\n  reachable fraction by distance from the base:")
    print(f"    {'band (mm)':>14}  {'n':>5}  {'reachable':>10}  {'median resid':>13}")
    edges = [0.40, 0.52, 0.60, 0.68, 0.76, 0.84, 0.95]
    for lo, hi in zip(edges, edges[1:]):
        ch = [r for r in rows if lo <= r["dist"] < hi]
        if not ch:
            continue
        res = [r["residual"] for r in ch if r["residual"] == r["residual"]]
        print(f"    {lo*1000:5.0f}-{hi*1000:5.0f}  {len(ch):5d}  "
              f"{sum(r['reachable'] for r in ch)/len(ch):10.3f}  "
              f"{np.median(res)*1000 if res else float('nan'):12.1f} mm")
    print("\n  A reachable fraction that falls off well before 855 mm is the")
    print("  point: the spec sphere is not the constraint, the frozen wrist is.")

    if any(r["manipulability"] == r["manipulability"] for r in rows):
        print(f"\n  manipulability sqrt(det(J J^T)) by band - even where IK")
        print(f"  succeeds, a poorly conditioned Jacobian means a delta_pos")
        print(f"  command of a given size buys less accurate Cartesian motion:")
        for lo, hi in zip(edges, edges[1:]):
            v = [r["manipulability"] for r in rows
                 if lo <= r["dist"] < hi and r["reachable"]
                 and r["manipulability"] == r["manipulability"]]
            if v:
                print(f"    {lo*1000:5.0f}-{hi*1000:5.0f}  median {np.median(v):.4f}")
    else:
        print("\n  (no manipulability column - the CPU Kinematics path has no")
        print("   pytorch_kinematics chain and the Pinocchio Jacobian call did")
        print("   not match this ManiSkill version. Not fatal.)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    os.makedirs(a.figdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    # Rows are generated x-major, y-minor, so reshape rather than matching on
    # float equality against the grid coordinates.
    R = np.array([r["reachable"] for r in rows],
                 dtype=float).reshape(len(xs), len(ys))
    ax.pcolormesh(xs, ys, R.T, cmap="RdYlGn", vmin=0, vmax=1, shading="nearest")
    bx, by = PANDA_BASE_XY
    ax.plot([bx], [by], marker="s", ms=9, c="k")
    # The sampler's support: cube centres can only ever land in here, so
    # reachability outside it is irrelevant however it looks.
    ax.add_patch(plt.Rectangle((-0.2, -0.3), 0.4, 0.6, fill=False, ec="k",
                               lw=1.4, ls="-"))
    ax.annotate("cube support", (-0.2, 0.30), fontsize=7, va="bottom")
    for rad, c in ((0.52, "darkorange"), (0.76, "tab:purple")):
        ax.add_patch(plt.Circle((bx, by), rad, fill=False, ls="--", ec=c, lw=1.2))
    ax.set_xlim(bx - 0.06, 0.26)
    ax.set_ylim(-0.38, 0.38)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Reachable with the frozen top-down gripper, z={a.z} m",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{a.figdir}/t2_reach_map.png", dpi=150)
    print(f"\nwrote {a.out} and {a.figdir}/t2_reach_map.png")


if __name__ == "__main__":
    main()
