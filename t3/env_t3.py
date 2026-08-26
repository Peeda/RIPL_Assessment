#!/usr/bin/env python
"""StackCube-T3-v1: StackCube with the LLM's reward and the LLM's sampler.

Importing this module registers the environment. Every T-III script that touches
a simulator imports it AT MODULE SCOPE, never inside a function - physx_cpu
vectorises by subprocess and forkserver re-imports __main__ in every child, so a
registration that happens inside main() does not exist in the workers and
gym.make dies there with a message about an unknown env id.

TWO ENVIRONMENT VARIABLES, AND THE SECOND ONE IS LOAD-BEARING
    T3_RUN      the generation directory holding reward.py and sampler.py.
    T3_SAMPLER  1 (default) to draw initial states from the generated sampler,
                0 to leave the environment's own _initialize_episode alone.

    T3_SAMPLER=0 is not a convenience. Every evaluation in this project
    addresses an episode by its seed, and t2/verify.py's strongest offline check
    joins the poses logged during a rollout against an independently built seed
    index. With the biased sampler active, reset(seed=s) no longer produces the
    state that seeds.csv records for s - by design - and that join fails on
    every episode. So:

        T3_SAMPLER=1   T-IV training only. A biased distribution is the point.
        T3_SAMPLER=0   anything that has to reproduce a T-II episode: the
                       alignment measurement, and T-IV's before/after scoring.

    It is an env var rather than an env kwarg because it has to survive the
    forkserver boundary and because t2/harness.py builds env_kwargs internally,
    which would otherwise need a second seam.

WHY SUBCLASSING, AND WHY super() FIRST
    ManiSkill has no hook for a custom reward: BaseEnv.get_reward dispatches on
    reward_mode to compute_dense_reward, and the way to supply one is to
    subclass and register. _initialize_episode calls super() BEFORE overriding
    the cube poses, so TableSceneBuilder.initialize still runs - it sets the
    table pose, the robot's initial qpos with its configured noise, and forces
    the gripper open. Reimplementing _initialize_episode wholesale would mean
    synthesising all of that, and getting the robot's start pose subtly wrong is
    exactly the kind of difference that would show up later as an unexplained
    gap between T-III's numbers and T-II's.
"""
import os
import sys

import torch

from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loader import load_file  # noqa: E402
from spec import REWARD_FILE, REWARD_MAX_NAME, SAMPLER_FILE  # noqa: E402

ENV_UID = "StackCube-T3-v1"

_CACHE = {}


def _run_dir():
    run = os.environ.get("T3_RUN")
    if not run:
        raise RuntimeError(
            "T3_RUN is not set. It must name the generation directory holding "
            f"{REWARD_FILE} and {SAMPLER_FILE} - e.g.\n"
            "    T3_RUN=$RIPL_ROOT/t3/gap bash t3/run.sh align")
    return run


def _load(kind):
    """The generated module, loaded once per process, through layer A.

    Static checks gate the import rather than the other way round: importing
    first would execute module-level code nobody has looked at yet.
    """
    key = (kind, _run_dir())
    if key not in _CACHE:
        fname = REWARD_FILE if kind == "reward" else SAMPLER_FILE
        path = os.path.join(_run_dir(), fname)
        if not os.path.exists(path):
            raise RuntimeError(f"{path} does not exist. Run `bash t3/run.sh "
                               f"generate` first.")
        _CACHE[key] = load_file(path, kind)
    return _CACHE[key]


def sampler_enabled():
    return os.environ.get("T3_SAMPLER", "1") not in ("0", "false", "False", "")


@register_env(ENV_UID, max_episode_steps=50)
class StackCubeT3Env(StackCubeEnv):
    """StackCube-v1 with a generated dense reward and a biased initial-state
    sampler. Everything else - the scene, the success criterion, the observation
    spaces, the controllers - is inherited unchanged, so a number measured here
    is comparable to a number measured on StackCube-v1.

    max_episode_steps matches the parent's registered default of 50; every
    caller in this repo overrides it to 200 through gym.make, exactly as they do
    for StackCube-v1.
    """

    # -- the sampler -------------------------------------------------------

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        if not sampler_enabled():
            return

        b = len(env_idx)
        with torch.device(self.device):
            out = _load("sampler")["sample_cube_poses"](b, self.device)

        # Validated here as well as in layer E, because layer E checks the
        # sampler in isolation and this is the only place that sees what the
        # environment actually received. A malformed batch that reached
        # set_pose would corrupt an entire training run silently.
        for key, shape in (("cubeA_xyz", (b, 3)), ("cubeA_quat", (b, 4)),
                           ("cubeB_xyz", (b, 3)), ("cubeB_quat", (b, 4))):
            t = out.get(key)
            if t is None:
                raise RuntimeError(f"sampler returned no '{key}' "
                                   f"(got {sorted(out)})")
            if tuple(t.shape) != shape:
                raise RuntimeError(f"sampler's '{key}' has shape "
                                   f"{tuple(t.shape)}, expected {shape}")
            if not torch.isfinite(t).all():
                raise RuntimeError(f"sampler's '{key}' contains non-finite values")

        dev = self.device
        self.cubeA.set_pose(Pose.create_from_pq(
            p=out["cubeA_xyz"].to(dev).float(), q=out["cubeA_quat"].to(dev).float()))
        self.cubeB.set_pose(Pose.create_from_pq(
            p=out["cubeB_xyz"].to(dev).float(), q=out["cubeB_quat"].to(dev).float()))

    # -- the reward --------------------------------------------------------

    @property
    def t3_reward_max(self):
        ns = _load("reward")
        return float(ns[REWARD_MAX_NAME])

    def compute_dense_reward(self, obs, action, info):
        return _load("reward")["compute_reward"](self, obs, action, info)

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / self.t3_reward_max


def describe():
    """One line naming what this process will actually run, for a log header.

    Printed by every stage. A pass that quietly used the wrong reward, or left
    the biased sampler on during a measurement that had to reproduce T-II
    episodes, is the failure this line exists to make impossible to miss.
    """
    run = os.environ.get("T3_RUN", "<unset>")
    return (f"env {ENV_UID}  T3_RUN={run}  "
            f"T3_SAMPLER={'on (biased)' if sampler_enabled() else 'off (nominal)'}")
