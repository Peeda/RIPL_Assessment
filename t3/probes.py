#!/usr/bin/env python
"""Layers B and C: is the reward well-formed, and is it hackable? Needs ManiSkill.

    python3 t3/probes.py --run t3/artifacts/gap --mode gap --out $T3_OUT
                         [--reward t3/fixtures/stock_reward.py] [--make-fixture CKPT]

LAYER B - shape, finiteness, purity, and that it does not touch the simulator.
    Cheap and unglamorous, and it catches the errors that would otherwise
    surface as a training run that diverges for no visible reason: a reward that
    returns a Python float and broadcasts, one that returns (n, 1) and turns the
    advantage into a matrix, one that divides by a distance that is zero exactly
    when the gripper reaches the cube.

    Two of these checks are worth more than they look.

    PURITY. Call the reward twice on an unchanged state and require bitwise
    equality. A single torch.rand_like buried in a shaping term makes every
    advantage estimate noisy in a way no amount of training averages out, and
    nothing else in the battery would notice.

    NO MUTATION. Snapshot env.get_state() before and after. The nastiest hack
    available here is a reward that moves cubeA onto cubeB itself and then
    collects the bonus for cubeA being on cubeB. Layer A bans .set_pose
    statically; this is the check that does not depend on having thought of the
    right method name.

LAYER C - hand-built states, and why they cannot be replaced by rollouts.
    The alignment measurement (layer D) only ever sees states the base policy
    actually produces. The policy never puts cubeB on top of cubeA, so no number
    of rollouts reveals a reward whose goal arguments are swapped; it never
    drops a cube through the table, so nothing reveals a reward that measures
    only horizontal distance. Those are mis-specifications rather than hacks,
    they are exactly what an LLM produces, and a hand-built state is the only
    thing that finds them.

    ONE STATE CANNOT BE INJECTED, AND IT IS THE MOST IMPORTANT ONE.
    Panda.is_grasping reads pairwise CONTACT FORCES from the last physics step
    (panda.py:225-253 -> scene.py:821-833). After a reset plus set_pose with no
    stepping there are no contacts, so is_cubeA_grasped is False in every
    injected state - and "cubeA held above cubeB and never released", the
    canonical consequence of a grasp-heavy reward, is precisely a grasped state.
    So P7 is restored from a state dict captured during a REAL rollout, stepped
    twice with the gripper closing to re-establish contact, and the flag is
    asserted to have come back. Discovering this inside a probe run costs an
    afternoon; it is written down here so it costs nothing.

THIS SCRIPT MEASURES; t3/verify.py DECIDES.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import gymnasium as gym  # noqa: E402
import mani_skill.envs  # noqa: F401,E402
import env_t3  # noqa: F401,E402  - registers StackCube-T3-v1
from loader import load_file  # noqa: E402
from mani_skill.utils.structs.pose import Pose  # noqa: E402
from spec import (PROBE_COLUMNS, PROBES, REWARD_FILE, REWARD_MAX_NAME,  # noqa: E402
                  SWEEP_COLUMNS, SWEEP_R, SWEEP_STEPS, SWEEP_Z)

GRASP_FIXTURE = os.path.join(HERE, "fixtures", "grasp_hover_states.npz")
PROBE_SEED = 11111
ALLOWED_ROOTS = {"num_envs", "device", "cube_half_size", "cubeA", "cubeB", "agent"}


class AttrSpy:
    """Records which top-level attributes of `env` the reward reads.

    Deliberately shallow. A fully nested recording proxy would have to wrap
    every Pose and Tensor it returns, and a wrapper that leaks into a torch op
    turns a validation tool into a source of bugs. Layer A already checks the
    full dotted chains statically; this closes the one hole layer A has - an
    alias (`c = env.cubeA` then `c.pose.p`) whose chain it cannot follow - by
    seeing the root that was actually touched.
    """

    def __init__(self, env):
        object.__setattr__(self, "_env", env)
        object.__setattr__(self, "touched", set())

    def __getattr__(self, name):
        object.__getattribute__(self, "touched").add(name)
        return getattr(object.__getattribute__(self, "_env"), name)


def make_env():
    return gym.make(
        "StackCube-T3-v1", num_envs=1, obs_mode="state", sim_backend="physx_cpu",
        reconfiguration_freq=0, reward_mode="sparse",
        control_mode=os.environ.get("CTRL", "pd_ee_delta_pos"),
        max_episode_steps=int(os.environ.get("MAX_EP_STEPS", 200)))


def _pose_of(actor):
    return actor.pose.p.clone(), actor.pose.q.clone()


def _set(actor, p, q=None):
    actor.set_pose(Pose.create_from_pq(p=p, q=q if q is not None else actor.pose.q))


def _score(reward_fn, env, obs, u):
    """The reward and evaluate()'s flags for whatever state the sim is in."""
    info = u.evaluate()
    action = torch.zeros((u.num_envs, env.action_space.shape[-1]))
    r = reward_fn(u, obs, action, info)
    return float(np.asarray(r.detach().cpu()).reshape(-1)[0]), info


# ---------------------------------------------------------------------------
# layer B
# ---------------------------------------------------------------------------


def layer_b(reward_fn, reward_max, env, seeds=(10001, 10002, 10003, 10004),
            steps=20):
    u = env.unwrapped
    res = dict(shape_ok=True, dtype_ok=True, finite_ok=True, pure_ok=True,
               mutation_ok=True, r_min=float("inf"), r_max=float("-inf"),
               n_calls=0, touched=[], bad_attrs=[], device_ok=True)
    spy = AttrSpy(u)

    for s in seeds:
        obs, _ = env.reset(seed=s)
        for t in range(steps):
            info = u.evaluate()
            action = env.action_space.sample()
            at = torch.as_tensor(np.asarray(action)).float().reshape(1, -1)

            r = reward_fn(spy, obs, at, info)
            res["n_calls"] += 1

            if not torch.is_tensor(r):
                res["shape_ok"] = res["dtype_ok"] = False
                res.setdefault("note", f"returned {type(r).__name__}, not a tensor")
                break
            if tuple(r.shape) != (u.num_envs,):
                res["shape_ok"] = False
                res.setdefault("note", f"shape {tuple(r.shape)} != ({u.num_envs},)")
            if not torch.is_floating_point(r):
                res["dtype_ok"] = False
            if str(r.device) != str(u.device):
                res["device_ok"] = False
            if not torch.isfinite(r).all():
                res["finite_ok"] = False
            v = r.detach().cpu().float().reshape(-1)
            res["r_min"] = min(res["r_min"], float(v.min()))
            res["r_max"] = max(res["r_max"], float(v.max()))

            # purity: identical state, identical answer
            r2 = reward_fn(spy, obs, at, info)
            if not torch.equal(r.detach(), r2.detach()):
                res["pure_ok"] = False

            # no mutation: the simulator is byte-identical after scoring
            before = u.get_state().clone()
            reward_fn(spy, obs, at, info)
            if not torch.allclose(before, u.get_state(), atol=0, rtol=0):
                res["mutation_ok"] = False

            obs, _, _, _, _ = env.step(action)

    res["touched"] = sorted(spy.touched)
    res["bad_attrs"] = sorted(spy.touched - ALLOWED_ROOTS)
    res["bounds_ok"] = (res["r_min"] >= -1e-5
                        and res["r_max"] <= reward_max + 1e-5)
    res["reward_max"] = reward_max
    return res


# ---------------------------------------------------------------------------
# layer C
# ---------------------------------------------------------------------------


def make_grasp_fixture(ckpt, out=GRASP_FIXTURE):
    """Roll the frozen base policy until it is holding cubeA well clear of the
    table, and save the whole simulator state.

    A few kB, committed (`.gitignore`'s `!*_states.npz` negation covers the
    name). It is the only way to get a GRASPED state into a probe, and without
    it the single most important ordering in the battery - a completed stack
    must beat a cube held above the target forever - cannot be tested at all.
    """
    sys.path.insert(1, os.path.join(os.path.dirname(HERE), "t2"))
    from harness import build_agent, to_device

    prev = os.environ.get("T3_SAMPLER")
    os.environ["T3_SAMPLER"] = "0"
    try:
        agent, envs, args, device = build_agent(
            ckpt, "--state" in sys.argv, 1,
            max_episode_steps=int(os.environ.get("MAX_EP_STEPS", 200)))
        for seed in range(10001, 10041):
            obs, info = envs.reset(seed=[seed])
            for _ in range(60):
                with torch.no_grad():
                    chunk = agent.get_action(to_device(obs, device)).cpu().numpy()
                for i in range(chunk.shape[1]):
                    obs, _, _, _, info = envs.step(chunk[:, i])
                    inner = envs.envs[0].unwrapped
                    if (bool(np.asarray(inner.evaluate()["is_cubeA_grasped"])
                             .reshape(-1)[0])
                            and float(inner.cubeA.pose.p[0, 2]) > 0.08):
                        os.makedirs(os.path.dirname(out), exist_ok=True)
                        np.savez(out,
                                 state=inner.get_state().cpu().numpy(),
                                 seed=np.array([seed]))
                        envs.close()
                        print(f"  saved grasp fixture from seed {seed} -> {out}")
                        return out
        envs.close()
    finally:
        if prev is None:
            os.environ.pop("T3_SAMPLER", None)
        else:
            os.environ["T3_SAMPLER"] = prev
    raise RuntimeError("no grasped-and-lifted state found in 40 episodes")


def _probe_states(env, obs, u):
    """Yield (probe_id, setup_fn). Each setup runs after a fresh reset."""
    env.reset(seed=PROBE_SEED)
    a0p, a0q = _pose_of(u.cubeA)
    b0p, b0q = _pose_of(u.cubeB)
    goal = b0p.clone()
    goal[:, 2] = b0p[:, 2] + 2 * float(u.cube_half_size[2])
    # A direction that walks cubeA away from cubeB while staying on the table.
    sx = -1.0 if float(b0p[0, 0]) > 0 else 1.0
    sy = -1.0 if float(b0p[0, 1]) > 0 else 1.0

    def s_success():
        _set(u.cubeA, goal.clone(), b0q.clone())

    def s_hover():
        p = goal.clone(); p[:, 2] += 0.12
        _set(u.cubeA, p, b0q.clone())

    def s_adjacent():
        p = b0p.clone(); p[:, 0] += sx * 0.05; p[:, 2] = 0.02
        _set(u.cubeA, p, b0q.clone())

    def s_knocked():
        p = b0p.clone(); p[:, 0] = sx * 0.18; p[:, 1] = sy * 0.28; p[:, 2] = 0.02
        _set(u.cubeB, p)

    def s_inverted():
        p = a0p.clone(); p[:, 2] = a0p[:, 2] + 2 * float(u.cube_half_size[2])
        _set(u.cubeB, p, a0q.clone())

    def s_offtable():
        p = goal.clone(); p[:, 2] = -0.10
        _set(u.cubeA, p, b0q.clone())

    def s_far():
        p = b0p.clone(); p[:, 1] += sy * 0.30; p[:, 2] = 0.02
        _set(u.cubeA, p, b0q.clone())

    def s_start():
        pass

    return dict(P0_success=s_success, P1_hover=s_hover, P2_adjacent=s_adjacent,
                P3_knocked=s_knocked, P4_inverted=s_inverted,
                P5_offtable=s_offtable, P6_far=s_far, P8_start=s_start), goal, (sx, sy)


def layer_c(reward_fn, env, reward_max):
    u = env.unwrapped
    obs, _ = env.reset(seed=PROBE_SEED)
    setups, goal, (sx, sy) = _probe_states(env, obs, u)

    rows, note = [], {}
    for pid, _desc in PROBES:
        if pid == "P7_held":
            continue
        obs, _ = env.reset(seed=PROBE_SEED)
        setups[pid]()
        r, info = _score(reward_fn, env, obs, u)
        rows.append(_probe_row(pid, r, reward_max, info, u))

    # --- P7: the grasped state, restored rather than injected --------------
    if os.path.exists(GRASP_FIXTURE):
        obs, _ = env.reset(seed=PROBE_SEED)
        state = torch.as_tensor(np.load(GRASP_FIXTURE)["state"]).float().to(u.device)
        u.set_state(state)
        # Two steps with the gripper closing and no translation, to re-establish
        # the finger contacts is_grasping reads. Without them the restored state
        # reports is_cubeA_grasped=False and the probe would be mislabelled.
        for _ in range(2):
            obs, _, _, _, _ = env.step(np.array([0.0, 0.0, 0.0, -1.0],
                                                dtype=np.float32))
        r, info = _score(reward_fn, env, obs, u)
        grasped = bool(np.asarray(info["is_cubeA_grasped"].cpu()).reshape(-1)[0])
        if grasped:
            rows.append(_probe_row("P7_held", r, reward_max, info, u))
        else:
            note["P7_held"] = ("the restored state did not report a grasp after "
                               "two closing steps - the fixture is stale, "
                               "regenerate it with --make-fixture")
    else:
        note["P7_held"] = (f"no grasp fixture at {GRASP_FIXTURE}; run "
                           f"`python3 t3/probes.py --make-fixture $CKPT`")

    # --- the two sweeps ----------------------------------------------------
    sweeps = []
    for name, lo, hi in (("z", *SWEEP_Z), ("r", *SWEEP_R)):
        for i in range(SWEEP_STEPS):
            off = lo + (hi - lo) * i / (SWEEP_STEPS - 1)
            obs, _ = env.reset(seed=PROBE_SEED)
            p = goal.clone()
            if name == "z":
                p[:, 2] = off
            else:
                p[:, 0] += sx * off
            _set(u.cubeA, p)
            r, _ = _score(reward_fn, env, obs, u)
            sweeps.append(dict(sweep=name, i=i, offset=off, reward=r))

    return rows, sweeps, note


def _probe_row(pid, r, reward_max, info, u):
    def f(k):
        return int(bool(np.asarray(info[k].cpu()).reshape(-1)[0]))
    a = u.cubeA.pose.p[0].tolist()
    b = u.cubeB.pose.p[0].tolist()
    return dict(probe=pid, reward=r, reward_norm=r / reward_max,
                is_cubeA_grasped=f("is_cubeA_grasped"),
                is_cubeA_on_cubeB=f("is_cubeA_on_cubeB"),
                is_cubeA_static=f("is_cubeA_static"), success=f("success"),
                cubeA_x=a[0], cubeA_y=a[1], cubeA_z=a[2],
                cubeB_x=b[0], cubeB_y=b[1], cubeB_z=b[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--mode", default="gap")
    ap.add_argument("--out")
    ap.add_argument("--reward", help="a reward file to test instead of the "
                                     "generation's - used for the stock-reward "
                                     "calibration arm")
    ap.add_argument("--label", help="name for the output files (default: mode)")
    ap.add_argument("--make-fixture", help="checkpoint; capture the grasp state "
                                           "and exit")
    ap.add_argument("--state", action="store_true")
    a = ap.parse_args()

    if a.make_fixture:
        make_grasp_fixture(a.make_fixture)
        return

    path = a.reward or os.path.join(a.run, REWARD_FILE)
    ns = load_file(path, "reward")
    reward_fn, reward_max = ns["compute_reward"], float(ns[REWARD_MAX_NAME])
    label = a.label or a.mode
    print(f"  {env_t3.describe()}")
    print(f"  reward      {path}  ({REWARD_MAX_NAME}={reward_max})")

    # Probes are hand-built states, so the biased sampler must be off - an
    # initial state drawn from the failure region would move cubeB out from
    # under every pose computed relative to it.
    os.environ["T3_SAMPLER"] = "0"
    env = make_env()
    try:
        b = layer_b(reward_fn, reward_max, env)
        print(f"  layer B     {b['n_calls']} calls   "
              f"shape {b['shape_ok']}  finite {b['finite_ok']}  "
              f"pure {b['pure_ok']}  no-mutation {b['mutation_ok']}  "
              f"range [{b['r_min']:.3f}, {b['r_max']:.3f}]")
        if b["bad_attrs"]:
            print(f"              !! read outside the surface: {b['bad_attrs']}")

        rows, sweeps, note = layer_c(reward_fn, env, reward_max)
    finally:
        env.close()

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"probes_{label}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PROBE_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(a.out, f"sweeps_{label}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SWEEP_COLUMNS)
        w.writeheader()
        w.writerows(sweeps)
    with open(os.path.join(a.out, f"probes_{label}.json"), "w") as f:
        json.dump(dict(mode=a.mode, label=label, reward=path,
                       reward_max=reward_max, layer_b=b, notes=note), f, indent=2)

    print("\n  layer C")
    for r in sorted(rows, key=lambda r: -r["reward"]):
        print(f"    {r['probe']:<12} {r['reward']:7.3f}  "
              f"({r['reward_norm']:.3f} of max)   "
              f"grasped={r['is_cubeA_grasped']} on_b={r['is_cubeA_on_cubeB']} "
              f"static={r['is_cubeA_static']} success={r['success']}")
    for k, v in note.items():
        print(f"    {k:<12} SKIPPED - {v}")
    print(f"\n  wrote       {a.out}/probes_{label}.{{csv,json}} and "
          f"sweeps_{label}.csv")
    print(f"  These are measurements. t3/verify.py applies the orderings.")


if __name__ == "__main__":
    main()
