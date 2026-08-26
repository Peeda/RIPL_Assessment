"""Layer A: no REWARD_MAX, so the environment cannot normalise the reward and
PPO gets an unbounded advantage signal."""
import torch


def compute_reward(env, obs, action, info):
    return torch.zeros(env.num_envs)
