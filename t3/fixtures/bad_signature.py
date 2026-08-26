"""Layer A: wrong arity - a plausible generation that read the ManiSkill
method signature rather than the contract."""
import torch

REWARD_MAX = 8.0


def compute_reward(env, info):
    return torch.zeros(env.num_envs)
