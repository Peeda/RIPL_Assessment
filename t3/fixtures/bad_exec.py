"""Layer A: builds code at run time. Nothing legitimate needs this."""
import torch

REWARD_MAX = 8.0


def compute_reward(env, obs, action, info):
    term = eval("1 + 1")
    return torch.zeros(env.num_envs) + term
