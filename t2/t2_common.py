#!/usr/bin/env python
"""Shared pieces for the T-II harness: angle conventions, cube features,
checkpoint loading, and the wrapper that gets ground truth out of subprocesses.

Exists so rollout_log.py (the localisation probe) and mine_rollouts.py (the
overnight miner) cannot drift apart on the two things that would silently
invalidate a comparison between them: the angle wrap and which weights get
loaded out of a checkpoint.

The one non-obvious piece is CubePoseInfo. rollout_log.py runs num_envs=1 so it
can reach env.unwrapped.cubeA directly; that is correct but caps throughput at
one episode at a time, which is too slow for the thousands of episodes T-II
needs. Under AsyncVectorEnv the env objects live in other processes and are not
reachable at all. CubePoseInfo closes that gap by pushing the ground truth into
info, which gymnasium already ships across the pipe.
"""
import math
import os
import subprocess
import sys

import numpy as np

# gymnasium and torch are needed for the rollout half of this module (the
# wrapper, checkpoint loading, env construction) and for nothing in the
# geometry half. analyze_rollouts.py and select_seeds.py are pure-CSV tools
# that import the geometry and must keep working on a laptop with no ManiSkill
# install - which is the whole reason the state arrays carry their own column
# names. So the sim imports are optional, and only the things that actually
# use them fail.
try:
    import gymnasium as gym
    import torch
    _Wrapper = gym.Wrapper
except ModuleNotFoundError:                          # analysis-only environment
    gym = torch = None
    _Wrapper = object

# --------------------------------------------------------------------------
# angles
# --------------------------------------------------------------------------


def wrap(a):
    """(-pi, pi]. The one convention; see CLAUDE.md. Applied at log time, once."""
    return -((-a + math.pi) % (2 * math.pi) - math.pi)


def wrap90(a):
    """(-pi/4, pi/4].

    A cube has 4-fold yaw symmetry, so a relative yaw of +85 deg and one of
    -5 deg describe the SAME geometry. CLAUDE.md pins relative_yaw to (-pi, pi]
    and that column is logged exactly as specified - but it is the wrong axis to
    regress against, because it splits one physical configuration across two
    ends of the range. This is the axis analysis should use.
    """
    q = math.pi / 2
    return -((-a + q / 2) % q - q / 2)


def yaw(q):
    """Yaw from a wxyz quaternion, wrapped."""
    w, x, y, z = [float(v) for v in np.asarray(q).reshape(-1)[:4]]
    return wrap(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


# --------------------------------------------------------------------------
# intervals
# --------------------------------------------------------------------------


def wilson(k, n, z=1.96):
    """Wilson score interval. Returns (lo, hi); (nan, nan) for n == 0.

    Not sqrt(p(1-p)/n): at n ~ 100 the interesting bins sit near p = 0, where
    the normal interval runs below zero and claims certainty it does not have.
    A 0/20 bin gets +-0.000 from the normal formula and [0, 0.161] from Wilson.
    Same data, honest bars. Shared so the analysis and the backend check cannot
    quote intervals computed two different ways.
    """
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


# --------------------------------------------------------------------------
# cube geometry - the T-II feature axes
# --------------------------------------------------------------------------

CUBE_FIELDS = (
    "cubeA_x", "cubeA_y", "cubeA_z", "cubeA_theta",
    "cubeB_x", "cubeB_y", "cubeB_z", "cubeB_theta",
    "separation", "relative_yaw", "relative_yaw_mod90",
)

# The Panda's base, from TableSceneBuilder.initialize:
# mani_skill/utils/scene_builder/table/scene_builder.py:103 (panda) and :123
# (panda_wristcam, which is StackCube's default robot) both do
#   self.env.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
# Every reach distance in T-II is measured from here. Getting this wrong once
# reversed the sign of a conclusion, so it is a named constant with a citation
# rather than an inline literal.
PANDA_BASE_XY = (-0.615, 0.0)

CUBE_HALF = 0.02                       # stack_cube.py:64,72 - 40 mm cubes

GEOM_FIELDS = (
    "face_gap", "dist_A", "dist_B", "dist_max", "dist_min", "far_is_B",
)


def _half_extent(delta, h=CUBE_HALF):
    """Half-width of an axis-aligned square of half-size h, measured along a
    direction `delta` radians from its own axes.

    h*(|cos|+|sin|): h at 0 deg, h*sqrt(2) at 45 deg. This is the whole reason
    yaw matters - a 40 mm cube presents 56.6 mm across its diagonal.
    """
    return h * (abs(math.cos(delta)) + abs(math.sin(delta)))


def geom_features(ax, ay, tha, bx, by, thb, h=CUBE_HALF):
    """Clearance and reach, from the two cube poses alone. -> GEOM_FIELDS dict.

    `face_gap` is the free space between the two cubes' FACES along the line
    joining their centres, where `separation` is the distance between the
    centres. The difference is the part that depends on yaw, and it is the part
    that predicts failure: within separation < 100 mm the low-gap half succeeds
    0.524 against the high-gap half's 0.670, so separation alone is discarding
    signal it had. It also explains why relative_yaw on its own measures
    nothing - yaw is only informative once combined with the A->B bearing,
    which is what enters here.

    `dist_*` are measured from PANDA_BASE_XY, not from the world origin. Success
    against dist_max is an inverted U (0.62 near, 0.83 mid, 0.55 far), so
    analysis must bin both tails rather than fitting a trend.
    """
    psi = math.atan2(by - ay, bx - ax)          # bearing from A to B
    sep = math.dist((ax, ay), (bx, by))
    dA = math.dist((ax, ay), PANDA_BASE_XY)
    dB = math.dist((bx, by), PANDA_BASE_XY)
    return dict(
        face_gap=sep - _half_extent(psi - tha, h) - _half_extent(psi + math.pi - thb, h),
        dist_A=dA, dist_B=dB,
        dist_max=max(dA, dB), dist_min=min(dA, dB),
        far_is_B=int(dB > dA),
    )


def cube_features(a_pose, b_pose):
    """Both raw_poses ([x,y,z,qw,qx,qy,qz]) -> CUBE_FIELDS + GEOM_FIELDS."""
    a = np.asarray(a_pose, dtype=float).reshape(-1)
    b = np.asarray(b_pose, dtype=float).reshape(-1)
    ta, tb = yaw(a[3:7]), yaw(b[3:7])
    rel = wrap(tb - ta)
    out = dict(
        cubeA_x=a[0], cubeA_y=a[1], cubeA_z=a[2], cubeA_theta=ta,
        cubeB_x=b[0], cubeB_y=b[1], cubeB_z=b[2], cubeB_theta=tb,
        separation=math.dist((a[0], a[1]), (b[0], b[1])),
        relative_yaw=rel,
        relative_yaw_mod90=wrap90(rel),
    )
    out.update(geom_features(a[0], a[1], ta, b[0], b[1], tb))
    return out


# The failure-mode regions, defined ONCE. verify.py asserts episodes satisfy
# them, demo_feasibility.py asks whether the planner can solve them, and
# run_modes.sh mines them - three consumers that must not drift apart. The
# thresholds are pre-registered; see notes/t2-failure-modes.md.
#
# gap and nearbase partition on dist_min, and both require dist_max < 0.76,
# which farb's dist_B >= 0.76 excludes - so the three are mutually exclusive
# and each controls for the others' factor.
REGIONS = {
    "gap":      lambda g: g["face_gap"] < 0.025 and 0.52 <= g["dist_min"] and g["dist_max"] < 0.76,
    "farb":     lambda g: g["dist_B"] >= 0.76 and g["dist_A"] < 0.72 and g["face_gap"] >= 0.05,
    "nearbase": lambda g: g["dist_min"] < 0.52 and g["dist_max"] < 0.76 and g["face_gap"] >= 0.05,
}

# What the nominal pass predicted for each, so a confirmation pass can be read
# as confirming or not rather than merely reported.
PREDICTED = {"gap": (0.640, 0.501, 0.759),
             "farb": (0.561, 0.410, 0.701),
             "nearbase": (0.637, 0.564, 0.704)}


def geom_from_row(r):
    """GEOM_FIELDS for a CSV row that predates them.

    Every column geom_features needs is already in the older CSVs, so the
    committed evidence base gains the new axes without being re-mined. Prefer
    the stored columns when a row already has them.
    """
    if all(k in r and r[k] != "" for k in GEOM_FIELDS):
        return {k: float(r[k]) for k in GEOM_FIELDS}
    return geom_features(float(r["cubeA_x"]), float(r["cubeA_y"]),
                         float(r["cubeA_theta"]),
                         float(r["cubeB_x"]), float(r["cubeB_y"]),
                         float(r["cubeB_theta"]))


# --------------------------------------------------------------------------
# the info wrapper
# --------------------------------------------------------------------------


def flatten_state_dict(d, prefix=""):
    """(names, values) from a ManiSkill state dict, walked in sorted-key order.

    Sorted rather than insertion order so the layout is stable across ManiSkill
    versions and across actor build order. Names travel with the values so the
    saved array is self-describing - the point of logging this at all is that
    the analysis reproduces without a working ManiSkill install.
    """
    names, vals = [], []
    if isinstance(d, dict):
        for k in sorted(d):
            n, v = flatten_state_dict(d[k], f"{prefix}/{k}" if prefix else k)
            names += n
            vals += v
    else:
        flat = np.asarray(torch.as_tensor(d).reshape(-1).float().cpu())
        names += [f"{prefix}[{i}]" for i in range(flat.shape[0])]
        vals += list(flat)
    return names, vals


def state_dict_from_flat(names, values):
    """Invert flatten_state_dict: (names, (N, D) array) -> ManiSkill state dict.

    Needed because the flat arrays in *_states.npz CANNOT be fed to
    env.set_state(). flatten_state_dict above walks in SORTED key order on
    purpose (stability across ManiSkill versions), while BaseEnv.set_state
    reconstructs by iterating self._init_raw_state["actors"] in INSERTION order
    (sapien_env.py:1316-1328). The two orders differ, and the mismatch is
    silent: set_state would happily read cubeB's pose out of cubeA's slot and
    the episode would just start somewhere unexpected.

    So round-trip through the names, which travel with the values precisely so
    the layout is self-describing, and hand back a dict.
    env.reset(options={"reset_to_env_states": {"env_states": <dict>}}) accepts
    one and routes to set_state_dict, which keys by name and cannot be
    misordered.
    """
    values = np.atleast_2d(np.asarray(values, dtype=np.float32))
    groups = {}                                  # (group, entity) -> [col idx]
    for col, nm in enumerate(np.asarray(names).reshape(-1)):
        head = str(nm).split("[")[0]             # "actors/cubeA[3]" -> "actors/cubeA"
        parts = head.split("/")
        if len(parts) != 2:
            raise ValueError(f"unexpected state name {nm!r}; expected 'group/entity[i]'")
        groups.setdefault((parts[0], parts[1]), []).append(col)

    out = {}
    for (group, entity), cols in groups.items():
        # Sorted-key order groups a given entity's columns contiguously and in
        # ascending index order, but do not rely on that - sort by the bracket.
        cols = sorted(cols, key=lambda c: int(str(np.asarray(names).reshape(-1)[c]).split("[")[1].rstrip("]")))
        # numpy when torch is absent: set_state_dict runs common.to_tensor
        # over what it is handed, so either works, and this keeps the
        # helper unit-testable off-pod.
        as_t = torch.as_tensor if torch is not None else np.asarray
        out.setdefault(group, {})[entity] = as_t(values[:, cols])
    return out


class CubePoseInfo(_Wrapper):
    """Push ground-truth poses into info so they survive the subprocess boundary.

    Applied via make_eval_envs(wrappers=[...]), so it sits INSIDE CPUGymWrapper.
    That is deliberate: CPUGymWrapper runs common.to_numpy then common.unbatch
    over the whole info dict, which turns the (1, 7) raw_pose tensors we add
    here into plain (7,) numpy arrays for free. Adding them outside would mean
    doing that conversion by hand.

    The full 70-float sim state is heavy to ship every step, so it is attached
    only where it is actually wanted: at reset (the initial state) and at the
    final step (the terminal state). Everything in between is covered by the
    three poses, which is what the trace needs.
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

    def _state(self):
        names, vals = flatten_state_dict(self.env.unwrapped.get_state_dict())
        return np.asarray(vals, dtype=np.float32), names

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        info.update(self._poses())
        st, names = self._state()
        info["env_state"] = st
        info["env_state_names"] = "|".join(names)
        info["episode_seed"] = int(np.asarray(self.env.unwrapped._episode_seed).reshape(-1)[0])
        return obs, info

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        info.update(self._poses())
        # elapsed_steps is what MSTimeLimit truncates on, so this is exactly the
        # last step of the episode - the state the success flags were judged in.
        if int(np.asarray(self.env.unwrapped.elapsed_steps).reshape(-1)[0]) >= self.max_episode_steps:
            info["env_state"] = self._state()[0]
        return obs, rew, term, trunc, info


# --------------------------------------------------------------------------
# checkpoints
# --------------------------------------------------------------------------

FLAG_KEYS = ("success_once", "success_at_end", "success",
             "is_cubeA_grasped", "is_cubeA_on_cubeB", "is_cubeA_static")


def flag(d, k):
    """One info entry as 0/1, or '' when the key is absent."""
    v = d.get(k) if isinstance(d, dict) else None
    return int(bool(np.asarray(v).reshape(-1)[0])) if v is not None else ""


def inspect_ckpt(path, state_mode):
    """Load a checkpoint and work out what it is, failing clearly if it is not
    what the caller said. Shared with rollout_log.py so both agree.

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


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------


def manifest(**extra):
    """Everything needed to interpret a number six weeks later.

    CLAUDE.md's rule: record the GPU model with every measurement, because
    numbers from different cards do not form a table. The PushT figures are
    missing that label and are worth less for it.
    """
    def sh(cmd, default="unknown"):
        try:
            return subprocess.check_output(cmd, shell=True, text=True,
                                           stderr=subprocess.DEVNULL).strip() or default
        except Exception:
            return default

    import gymnasium

    def sha(path, n=1 << 20):
        """Content hash of the checkpoint, so a number is attributable to the
        exact weights that produced it. The path alone is not enough:
        train_rgbd.py derives its output dir from --exp-name and overwrites
        checkpoints in place, so the same path can hold different weights on
        different days."""
        import hashlib
        try:
            h = hashlib.sha256()
            with open(path, "rb") as f:
                while (b := f.read(n)):
                    h.update(b)
            return h.hexdigest()[:16]
        except Exception:
            return "unknown"

    # -C the repo explicitly: run_t2.sh cds into the output dir before invoking
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
    return m


def build_agent(ckpt_path, state_mode, num_envs, device=None, video_dir=None,
                max_episode_steps=200, expose_poses=True):
    """The full checkpoint -> (agent, envs, args) path, shared by every script.

    Centralised because the pool_feature_map handling below is easy to get
    subtly wrong in a way that does not fail loudly, and having two copies of it
    is how the rgb and state numbers would end up non-comparable.
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
        import dataclasses
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

    env_kwargs = dict(control_mode=ctrl, reward_mode="sparse", obs_mode=obs_mode,
                      render_mode="rgb_array",
                      human_render_camera_configs=dict(shader_pack="default"),
                      max_episode_steps=max_episode_steps)
    envs = make_eval_envs(env_id, num_envs, backend, env_kwargs,
                          dict(obs_horizon=args.obs_horizon),
                          video_dir=video_dir, wrappers=wrappers)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = load_weights(T.Agent(envs, args).to(device), sd)
    return agent, envs, args, device


def to_device(obs, device):
    """Observations come back as numpy (physx_cpu) in either dict or array form."""
    if isinstance(obs, dict):
        return {k: torch.as_tensor(np.asarray(v)).to(device) for k, v in obs.items()}
    return torch.as_tensor(np.asarray(obs)).to(device)
