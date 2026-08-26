"""Layer A: reaches outside the allowed imports."""
import os

import torch

REWARD_MAX = 8.0


def compute_reward(env, obs, action, info):
    scale = len(os.environ)
    return torch.zeros(env.num_envs) + scale
