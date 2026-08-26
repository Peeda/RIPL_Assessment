#!/usr/bin/env python
"""The simulator half: checkpoint loading, env construction, ground-truth access.

Split from t2/geometry.py, which defines what a failure mode IS and imports
nothing but `math`. Everything here needs gymnasium, torch and a ManiSkill
install, so it imports them unconditionally and fails loudly off-pod rather than
degrading into a half-working module.

The one non-obvious piece is CubePoseInfo. A single-process env can reach
`env.unwrapped.cubeA` directly, but physx_cpu vectorises by SUBPROCESS, so at
any useful width the env objects live in other processes and are not reachable
at all. CubePoseInfo pushes the ground truth into `info`, which gymnasium
already ships across the pipe.
"""
import os
import subprocess
import sys

import gymnasium as gym
import numpy as np
import torch

# t4/residual.py imports torch and nothing else, so this costs nothing and is
# safe off-pod. It is imported unconditionally because build_agent ALWAYS wraps:
# see the ResidualAgent block below.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "t4"))
from residual import ResidualAgent, load_residual  # noqa: E402

# ---------------------------------------------------------------------------
# ground truth across the subprocess boundary
# ---------------------------------------------------------------------------


class CubePoseInfo(gym.Wrapper):
    """Push ground-truth poses into info so they survive the subprocess boundary.

    Applied via make_eval_envs(wrappers=[...]), so it sits INSIDE CPUGymWrapper.
    That is deliberate: CPUGymWrapper runs common.to_numpy then common.unbatch
    over the whole info dict, which turns the (1, 7) raw_pose tensors added here
    into plain (7,) numpy arrays for free. Adding them outside would mean doing
    that conversion by hand.

    `episode_seed` is exposed for the same reason: eval_modes.py asserts the env
    reset to the seed that was asked for, and that assertion is only possible if
    the value crosses the pipe.
    """

    def __init__(self, env, max_episode_steps=200):
        super().__init__(env)
        self.max_episode_steps = max_episode_steps

    def _poses(self):
        u = self.env.unwrapped
        return dict(
            cubeA_pose=u.cubeA.pose.raw_pose,
            cubeB_pose=u.cubeB.pose.raw_pose,
            tcp_pose=u.agent.tcp.pose.raw_pose,
        )

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        info.update(self._poses())
        info["episode_seed"] = int(
            np.asarray(self.env.unwrapped._episode_seed).reshape(-1)[0])
        return obs, info

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        info.update(self._poses())
        return obs, rew, term, trunc, info


def poses_from_info(info, key, n):
    """The n per-env poses under `info[key]`, as a list of plain float lists.

    The vector env has already stacked info into per-env arrays, so this reads
    them positionally. Returned as lists rather than arrays because everything
    downstream is geometry.py, which takes sequences and imports no numpy.
    """
    return [[float(v) for v in np.asarray(p, float).reshape(-1)]
            for p in np.atleast_1d(info[key])[:n]]


def flag(d, k):
    """One info entry as 0/1, or '' when the key is absent."""
    v = d.get(k) if isinstance(d, dict) else None
    return int(bool(np.asarray(v).reshape(-1)[0])) if v is not None else ""


# ---------------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------------


def inspect_ckpt(path, state_mode):
    """Load a checkpoint and work out what it is, failing clearly if it is not
    what the caller said.

    Returns (state_dict, is_rgb, pooled). pooled is None for state checkpoints.
    """
    sd = torch.load(path, map_location="cpu")
    keys = (sd.get("ema_agent") or sd.get("agent")).keys()
    is_rgb = any(k.startswith("visual_encoder.") for k in keys)
    name = os.path.basename(path)
    if is_rgb and state_mode:
        sys.exit(f"\n{name} is an RGB checkpoint (it has visual_encoder weights) "
                 f"but --state was passed.\nDrop --state, or point at the state "
                 f"run's checkpoint.\n")
    if not is_rgb and not state_mode:
        sys.exit(f"\n{name} is a state checkpoint (no visual_encoder weights) but "
                 f"--state was not passed.\nAdd --state, or point at the rgb "
                 f"run's checkpoint.\n")
    print(f"  checkpoint mode: {'rgb' if is_rgb else 'state'}  (matches args)")

    # Infer the encoder variant from the weights rather than trusting a flag the
    # caller has to remember. visual_encoder.fc.0.weight is (256, 128) when
    # PlainConv global-max-pooled its feature map and (256, 8192) when it kept
    # the 8x8 grid. Getting this wrong is a shape mismatch at load_state_dict,
    # and the message names 8192 vs 128 rather than the cause.
    pooled = None
    if is_rgb:
        w = (sd.get("ema_agent") or sd.get("agent"))["visual_encoder.fc.0.weight"]
        pooled = w.shape[1] == 128
        print(f"  visual encoder:  "
              f"{'pooled (global max, no spatial map)' if pooled else 'spatial (8x8 map kept)'}"
              f"  [fc in={w.shape[1]}]")
    return sd, is_rgb, pooled


def load_weights(agent, sd):
    """ema_agent is what train.py evaluates with, so it is what produced the
    reported success numbers. Prefer it over the raw agent weights."""
    agent.load_state_dict(sd.get("ema_agent") or sd.get("agent"))
    agent.eval()
    return agent


def residual_path(block=None):
    """The residual checkpoint for this run, or None.

    $RESIDUAL may contain '{block}', because T-IV pairs residual seed b with
    evaluation block b: the three blocks are three independent training runs,
    so the reported spread carries training variance and not just DDPM
    sampling noise. With no '{block}' the same head is used throughout, and
    with RESIDUAL unset there is no residual at all.
    """
    p = os.environ.get("RESIDUAL", "").strip()
    if not p:
        return None
    if "{block}" in p:
        if block is None:
            return None                      # resolved later, per block
        p = p.format(block=block)
    if not os.path.exists(p):
        sys.exit(f"\nRESIDUAL={p} does not exist.\n"
                 f"Train it first, or unset RESIDUAL to evaluate the base "
                 f"policy.\n")
    return p


def attach_residual(agent, path, device):
    """Load `path` into `agent` and say so. Returns the resolved metadata."""
    head, alpha, act_horizon, meta = load_residual(path, device)
    if act_horizon != agent.act_horizon:
        sys.exit(f"\n{os.path.basename(path)} was trained against "
                 f"act_horizon={act_horizon} but this checkpoint's is "
                 f"{agent.act_horizon}.\nThe base policy would be re-planned at "
                 f"a different rate than it was trained and evaluated at.\n")
    agent.set_residual(head, alpha)
    print(f"  residual:        {os.path.basename(path)}  alpha={alpha:.4f} "
          f"({alpha * 100:.1f} mm/step)  res_horizon={head.res_horizon}  "
          f"mode={meta.get('mode', '?')} seed={meta.get('seed', '?')}")
    return meta


def build_agent(ckpt_path, state_mode, num_envs, device=None, video_dir=None,
                max_episode_steps=200, expose_poses=True, reward_mode="sparse"):
    """The full checkpoint -> (agent, envs, args, device) path, shared by every
    script that runs a policy.

    Centralised because the pool_feature_map handling below is easy to get
    subtly wrong in a way that does not fail loudly, and having two copies of it
    is how two numbers end up non-comparable.
    """
    dp = f"{os.environ['MANISKILL_REPO']}/examples/baselines/diffusion_policy"
    if dp not in sys.path:
        sys.path.insert(0, dp)
    from diffusion_policy.make_env import make_eval_envs

    sd, is_rgb, pooled = inspect_ckpt(ckpt_path, state_mode)

    if state_mode:
        import train as T
        obs_mode, wrappers = "state", []
    else:
        from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
        import train_rgbd as T
        obs_mode, wrappers = "rgb", [FlattenRGBDObservationWrapper]

    env_id = os.environ.get("ENV_ID", "StackCube-v1")
    ctrl = os.environ.get("CTRL", "pd_ee_delta_pos")
    backend = os.environ.get("BACKEND", "physx_cpu")

    kw = dict(env_id=env_id, demo_path="unused.h5", control_mode=ctrl,
              sim_backend=backend, max_episode_steps=max_episode_steps)
    if not state_mode:
        kw["obs_mode"] = "rgb"
        # patches/0001 makes this an Args field defaulting to False. On a stock
        # checkout the encoder is hardwired to pooled and there is nothing to
        # set - which is fine for a pooled checkpoint and fatal for a spatial
        # one, so say which it is rather than letting load_state_dict report
        # 8192 vs 128 and leave the cause unnamed.
        import dataclasses
        if any(f.name == "pool_feature_map" for f in dataclasses.fields(T.Args)):
            kw["pool_feature_map"] = pooled
        elif not pooled:
            sys.exit(f"\nThis is a spatial-encoder checkpoint but "
                     f"{os.environ['MANISKILL_REPO']} is stock upstream.\n"
                     f"Run 'bash setup/apply_patches.sh' first.\n")
    args = T.Args(**kw)

    if expose_poses:
        wrappers = wrappers + [lambda e: CubePoseInfo(e, max_episode_steps)]

    # reward_mode defaults to sparse, which is what every T-II number was
    # measured under and must stay. T-III overrides it to "dense" so the env
    # returns the LLM-generated reward and CPUGymWrapper's record_metrics sums
    # it into info['episode']['return'] for free - which is how the alignment
    # measurement gets cumulative reward without a second rollout loop.
    env_kwargs = dict(control_mode=ctrl, reward_mode=reward_mode, obs_mode=obs_mode,
                      render_mode="rgb_array",
                      human_render_camera_configs=dict(shader_pack="default"),
                      max_episode_steps=max_episode_steps)
    envs = make_eval_envs(env_id, num_envs, backend, env_kwargs,
                          dict(obs_horizon=args.obs_horizon),
                          video_dir=video_dir, wrappers=wrappers)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = load_weights(T.Agent(envs, args).to(device), sd)

    # ALWAYS wrapped, even with no residual. CLAUDE.md: "Both arms must go
    # through that one path, or the before/after compares two code paths as
    # well as two policies." With head=None ResidualAgent adds no tensor op and
    # returns the base's own chunk, so the before arm is bit-identical to what
    # every committed T-II number was measured on.
    agent = ResidualAgent(base, head=None, act_horizon=args.act_horizon)
    p = residual_path()
    if p:
        attach_residual(agent, p, device)
    elif os.environ.get("RESIDUAL", "").strip():
        print(f"  residual:        per-block, from "
              f"{os.environ['RESIDUAL']}")
    return agent, envs, args, device


def to_device(obs, device):
    """Observations -> torch on `device`, whatever backend produced them.

    physx_cpu hands back numpy through CPUGymWrapper; physx_cuda hands back CUDA
    tensors from ManiSkillVectorEnv, and `np.asarray` on one of those RAISES
    ("can't convert cuda:0 device type tensor to numpy"). So tensors pass
    through untouched and only non-tensors go via numpy. The CPU path is
    unchanged; this is what makes t4/backend_check.py able to run at all.
    """
    if isinstance(obs, dict):
        return {k: to_device(v, device) for k, v in obs.items()}
    if isinstance(obs, torch.Tensor):
        return obs.to(device)
    return torch.as_tensor(np.asarray(obs)).to(device)


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------


def sha16(path, n=1 << 20):
    """Content hash of a file, truncated to 16 hex chars.

    Module level because the per-block residual is hashed by eval_modes.py,
    after manifest() has already run, and two implementations of "the hash"
    is exactly how two numbers stop being attributable to the same weights.
    """
    import hashlib
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while (b := f.read(n)):
                h.update(b)
        return h.hexdigest()[:16]
    except Exception:
        return "unknown"


def manifest(**extra):
    """Everything needed to interpret a number six weeks later.

    CLAUDE.md's rule: record the GPU model with every measurement, because
    numbers from different cards do not form a table.
    """
    def sh(cmd, default="unknown"):
        try:
            return subprocess.check_output(cmd, shell=True, text=True,
                                           stderr=subprocess.DEVNULL).strip() or default
        except Exception:
            return default

    # sha16 is module level; see its docstring. The path alone is not enough
    # to attribute a number, because train_rgbd.py overwrites checkpoints in
    # place, so the same path can hold different weights on different days.
    # verify.py asserts every pass shares one hash.
    sha = sha16

    import gymnasium

    # -C the repo explicitly: the driver cds into the output dir before invoking
    # these scripts, so a bare `git rev-parse` would describe whatever repo the
    # output happens to sit in, or nothing at all.
    repo = os.path.dirname(os.path.abspath(__file__))
    m = dict(
        gpu=sh("nvidia-smi --query-gpu=name --format=csv,noheader | head -1"),
        torch=torch.__version__,
        gymnasium=gymnasium.__version__,
        repo_sha=sh(f"git -C {repo} rev-parse --short HEAD"),
        repo_dirty=sh(f"git -C {repo} status --porcelain") != "",
        maniskill_sha=sh(f"git -C {os.environ.get('MANISKILL_REPO', '.')} rev-parse --short HEAD"),
        env_id=os.environ.get("ENV_ID", "StackCube-v1"),
        control_mode=os.environ.get("CTRL", "pd_ee_delta_pos"),
        backend=os.environ.get("BACKEND", "physx_cpu"),
    )
    m.update(extra)
    if "ckpt" in m:
        m["ckpt_sha256"] = sha(m["ckpt"])
    # Same argument as ckpt_sha256: a residual path is not enough to attribute a
    # number, because training overwrites in place. Only present when a residual
    # was actually used, so every pre-T-IV manifest is unchanged.
    if m.get("residual"):
        m["residual_sha256"] = sha(m["residual"])
    return m
