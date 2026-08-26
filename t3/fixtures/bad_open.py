"""Layer A: touches the filesystem from inside a reward."""
import torch

REWARD_MAX = 8.0


def compute_reward(env, obs, action, info):
    f = open("/tmp/reward.log", "a")
    return torch.zeros(env.num_envs)
