#!/usr/bin/env python
"""The training environment, matching the base policy's observation stack.

The wrapper order is NOT a choice - the frozen base was trained and measured
against exactly one observation pipeline, and a residual trained against a
different one would be correcting a policy that does not exist.

TWO BACKENDS, AND WHY THE DEFAULT IS THE SLOW ONE
-------------------------------------------------

t4/backend_check.py replayed 300 identical physx_cpu initial states on
physx_cuda with the frozen base and measured:

    success_once   0.730 (cpu)  ->  0.557 (gpu)     -0.173
    agreement      0.667   against a ~0.74 same-backend floor
    McNemar        chi2=26.0, p<0.001   DIRECTIONAL SHIFT

    grasped        0.973  ->  0.987     +0.013
    placed         0.860  ->  0.783     -0.077
    success|placed 0.849  ->  0.711     -0.138

Perception is fine - the grasp rate HOLDS, so the visual policy is finding the
cube and picking it up, and a rendering difference would have shown there. The
loss is contact physics, concentrated after placement. It is not a settable
difference either: SceneConfig's solver iterations are shared, sapien_env.py
does not branch on backend, and StackCube does not override
_default_sim_config.

That lands precisely on the stage T-II's `farb` mode is DEFINED by ("places at
near-baseline rates, then the stack does not settle"). A residual trained on
GPU would be learning to fix a backend artifact, and neither a positive nor a
null result could be attributed to the method. So training runs on physx_cpu,
where every other number in this project was measured.

physx_cpu raises RuntimeError for num_envs > 1 - it vectorises by SUBPROCESS -
so the CPU branch delegates to the diffusion-policy baseline's own
make_eval_envs. That is deliberate beyond laziness: the training env is then
constructed by literally the same function t2/harness.py builds the scoring env
with, so the two pipelines cannot drift.

Forkserver re-imports __main__ in every child, which is why train_ppo.py keeps
`import env_t4` at module scope and its work under `if __name__ == "__main__"`.
Without that the workers do not know the env id and the parent reports a reset
socket (CLAUDE.md, Known traps).
"""
import os
import sys

import gymnasium as gym

from mani_skill.utils.wrappers import FrameStack
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


def make_train_envs(env_id, num_envs, obs_horizon=2, max_episode_steps=200,
                    control_mode="pd_ee_delta_pos", obs_mode="rgb",
                    reward_mode="normalized_dense", sim_backend="physx_cpu",
                    reconfiguration_freq=None, ignore_terminations=True):
    """-> a vector env. On physx_cpu it is an AsyncVectorEnv of subprocesses;
    on physx_cuda a batched ManiSkillVectorEnv. Both deliver the same
    observation dict, and train_ppo.py normalises the two info conventions."""
    env_kwargs = dict(control_mode=control_mode, reward_mode=reward_mode,
                      obs_mode=obs_mode, render_mode="rgb_array",
                      human_render_camera_configs=dict(shader_pack="default"),
                      max_episode_steps=max_episode_steps)

    if sim_backend == "physx_cpu":
        dp = f"{os.environ['MANISKILL_REPO']}/examples/baselines/diffusion_policy"
        if dp not in sys.path:
            sys.path.insert(0, dp)
        from diffusion_policy.make_env import make_eval_envs
        wrappers = [] if obs_mode == "state" else [FlattenRGBDObservationWrapper]
        # No video_dir: RecordEpisode attaches to sub-env 0 only, and a
        # 200-step mp4 every episode for the whole run is pure cost.
        return make_eval_envs(env_id, num_envs, sim_backend, env_kwargs,
                              dict(obs_horizon=obs_horizon), wrappers=wrappers)

    env = gym.make(env_id, num_envs=num_envs, sim_backend=sim_backend,
                   reconfiguration_freq=reconfiguration_freq, **env_kwargs)
    if obs_mode != "state":
        env = FlattenRGBDObservationWrapper(env)
    env = FrameStack(env, num_stack=obs_horizon)
    return ManiSkillVectorEnv(env, num_envs,
                              ignore_terminations=ignore_terminations,
                              record_metrics=True)
