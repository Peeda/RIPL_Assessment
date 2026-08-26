#!/usr/bin/env python
"""StackCube-T4-v1: T-III's reward and sampler, with a nominal-mixing knob.

Importing this module registers the environment. Import it AT MODULE SCOPE for
the same reason t3/env_t3.py says to: physx_cpu vectorises by subprocess and
forkserver re-imports __main__ in every child, so a registration inside main()
does not exist in the workers.

    T3_RUN            the generation directory holding reward.py and sampler.py
    T3_SAMPLER        1 (default) biased initial states, 0 the env's own
    T4_NOMINAL_FRAC   0.0 (default) what fraction of episodes bypass the
                      biased sampler and use StackCube's own placement

WHY THE MIXING KNOB EXISTS, AND WHY IT DEFAULTS TO OFF
------------------------------------------------------

T-III's sampler concentrates ~90-99% of episodes in one failure region, against
a nominal base rate of 3-5%. That is the point: almost every PPO gradient step
is then about the thing we want fixed. But the residual it produces fires on
EVERY state at evaluation, including the ~95% of nominal episodes that look
nothing like training - which is the mechanism by which a targeted fine-tune
degrades general performance, and the assignment explicitly asks for near-zero
degradation.

Policy Decorator's defence is the bounded residual alone: alpha caps the
per-step correction at ~5 mm, so the damage is bounded but not zero. Mixing is
the second line, and it is left OFF by default because "the episodic
configuration from T-III" is what the assignment specifies. Raise it only if
the nominal arm actually degrades, and then report both runs - the comparison
is a result either way ("the bound was sufficient" / "it was not").

With T4_NOMINAL_FRAC=0 this class is behaviourally identical to
StackCube-T3-v1; the blending branch is not even entered.

NOTE: register_env warns-and-skips on a duplicate uid (registration.py:221-230),
so editing this file does not take effect in a live process. Restart it.
"""
import os
import sys

import torch

from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "t3"))
from env_t3 import StackCubeT3Env, _load, sampler_enabled  # noqa: E402

ENV_UID = "StackCube-T4-v1"


def nominal_frac():
    v = float(os.environ.get("T4_NOMINAL_FRAC", "0") or 0)
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"T4_NOMINAL_FRAC={v} is outside [0, 1]")
    return v


def _rows(t, env_idx, b):
    """`t` restricted to the b rows this reset is initialising.

    ManiSkill masks the scene during _initialize_episode, so reading an actor's
    pose back may hand over either the b masked rows or all num_envs of them
    depending on version and backend. Rather than assume, take whichever is
    consistent and fail loudly on anything else - silently mixing one env's
    cube pose into another's initial state is precisely the kind of wrongness
    that produces a plausible CSV.
    """
    if t.shape[0] == b:
        return t
    if t.shape[0] > b:
        return t[env_idx]
    raise RuntimeError(
        f"pose read-back has {t.shape[0]} rows but this reset is initialising "
        f"{b} envs; cannot align them. T4_NOMINAL_FRAC>0 is unsafe on this "
        f"ManiSkill version - run the primary T4_NOMINAL_FRAC=0 arm instead.")


@register_env(ENV_UID, max_episode_steps=50)
class StackCubeT4Env(StackCubeT3Env):
    """T-III's env plus per-row mixing. max_episode_steps matches the parent's
    registered default of 50; every caller overrides it to 200 via gym.make."""

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        frac = nominal_frac()
        if frac <= 0.0:
            # The common path, and the one the primary runs take: exactly
            # StackCube-T3-v1, with no extra tensor touched.
            return super()._initialize_episode(env_idx, options)

        # Nominal placement for every row first - this also sets the table pose
        # and the robot's start qpos, which is why env_t3 calls super() first
        # too and why neither of us reimplements it.
        StackCubeEnv._initialize_episode(self, env_idx, options)
        if not sampler_enabled() or frac >= 1.0:
            return

        b = len(env_idx)
        with torch.device(self.device):
            out = _load("sampler")["sample_cube_poses"](b, self.device)
            # Biased where the draw says so. Drawn from torch under the env's
            # own forked, seeded generator (sapien_env.py:951), so the mix is
            # reproducible with everything else about the episode.
            biased = (torch.rand(b) >= frac)

        dev = self.device
        for actor, kxyz, kquat in ((self.cubeA, "cubeA_xyz", "cubeA_quat"),
                                   (self.cubeB, "cubeB_xyz", "cubeB_quat")):
            p = _rows(actor.pose.p, env_idx, b).clone()
            q = _rows(actor.pose.q, env_idx, b).clone()
            m = biased.to(dev).unsqueeze(-1)
            p = torch.where(m, out[kxyz].to(dev).float(), p)
            q = torch.where(m, out[kquat].to(dev).float(), q)
            actor.set_pose(Pose.create_from_pq(p=p, q=q))


def describe():
    """One line naming what this process will actually run, for a log header."""
    run = os.environ.get("T3_RUN", "<unset>")
    f = nominal_frac()
    return (f"env {ENV_UID}  T3_RUN={run}  "
            f"T3_SAMPLER={'on (biased)' if sampler_enabled() else 'off (nominal)'}  "
            f"T4_NOMINAL_FRAC={f:g}"
            f"{'  [pure T-III distribution]' if f == 0 else ''}")
