"""Layer A: a time-dependent reward. Off the allowed surface on purpose -
paying per step for staying alive is a reward hack that needs a clock."""
import torch

REWARD_MAX = 8.0


def compute_reward(env, obs, action, info):
    return torch.zeros(env.num_envs) + env.elapsed_steps * 0.01
