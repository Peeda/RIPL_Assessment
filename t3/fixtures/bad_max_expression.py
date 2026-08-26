"""Layer A: REWARD_MAX is an expression, so the gate cannot read it back out of
the file without importing the module it has not checked yet."""
import torch

REWARD_MAX = 4.0 * 2


def compute_reward(env, obs, action, info):
    return torch.zeros(env.num_envs)
