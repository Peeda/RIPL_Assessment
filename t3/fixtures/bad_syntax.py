"""Layer A: does not parse. An LLM truncated by max_tokens looks exactly like this."""
import torch

REWARD_MAX = 8.0


def compute_reward(env, obs, action, info):
    reward = torch.zeros(env.num_envs
    return reward
