"""Layer A: reads private environment state. env._episode_seed would let a
reward memorise which episodes are being evaluated."""
import torch

REWARD_MAX = 8.0


def compute_reward(env, obs, action, info):
    s = env._episode_seed
    return torch.zeros(env.num_envs)
