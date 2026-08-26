#!/usr/bin/env python
"""The GPU training environment, matching the base policy's observation stack.

The wrapper order here is NOT a choice - it is copied from the GPU branch of
the diffusion-policy baseline's make_env.py, which is what t2/harness.py builds
the evaluation env with. The frozen base policy was trained and measured
against exactly this observation pipeline; a residual trained against a
different one would be learning to correct a policy that does not exist.

    FlattenRGBDObservationWrapper   sensor_data -> {"rgb", "state"}
    FrameStack(obs_horizon)         -> (B, obs_horizon, ...)
    ManiSkillVectorEnv              torch tensors, batched, on device

Two deliberate differences from the eval path, both stated so neither looks
like drift:

  reconfiguration_freq=None   The eval path uses 1, which rebuilds the scene on
                              every reset. StackCube's geometry never changes -
                              two 40 mm cubes and a table - so on GPU that is
                              pure cost. It changes nothing about the
                              initial-state distribution, which comes from
                              _initialize_episode.

  reward_mode                 "normalized_dense" rather than the eval path's
                              "sparse". env_t3.compute_normalized_dense_reward
                              divides the generated reward by its own
                              REWARD_MAX, so a per-step reward is already in
                              [0, 1] and PPO needs no reward scaling. The
                              contract in t3/spec.py was written for this.

ignore_terminations=True is the same on both paths, and it is load-bearing for
PPO: every episode runs the full 200 steps, so at act_horizon 8 an episode is
exactly 25 chunked timesteps with no ragged remainder, all envs truncate
together, and GAE never has to cross an episode boundary.
"""
import gymnasium as gym

from mani_skill.utils.wrappers import FrameStack
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


def make_train_envs(env_id, num_envs, obs_horizon=2, max_episode_steps=200,
                    control_mode="pd_ee_delta_pos", obs_mode="rgb",
                    reward_mode="normalized_dense", sim_backend="physx_cuda",
                    reconfiguration_freq=None, ignore_terminations=True):
    env = gym.make(env_id, num_envs=num_envs, sim_backend=sim_backend,
                   reconfiguration_freq=reconfiguration_freq,
                   obs_mode=obs_mode, control_mode=control_mode,
                   reward_mode=reward_mode, render_mode="rgb_array",
                   human_render_camera_configs=dict(shader_pack="default"),
                   max_episode_steps=max_episode_steps)
    if obs_mode != "state":
        env = FlattenRGBDObservationWrapper(env)
    env = FrameStack(env, num_stack=obs_horizon)
    return ManiSkillVectorEnv(env, num_envs, ignore_terminations=ignore_terminations,
                              record_metrics=True)
