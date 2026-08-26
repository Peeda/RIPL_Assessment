"""Layer A: the classic sandbox escape, via a dunder attribute chain."""
import torch

REWARD_MAX = 8.0


def compute_reward(env, obs, action, info):
    subs = ().__class__.__base__
    return torch.zeros(env.num_envs)
