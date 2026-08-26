"""Layer A: right arity, WRONG ORDER. This one parses, imports, runs, and
silently computes the reward from the wrong tensor - which is why the checker
compares parameter names in order rather than counting them."""
import torch

REWARD_MAX = 8.0


def compute_reward(env, action, obs, info):
    return torch.zeros(env.num_envs)
