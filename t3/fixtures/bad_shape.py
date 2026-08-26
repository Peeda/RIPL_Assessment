"""Layer B: returns (num_envs, 1) instead of (num_envs,). Broadcasts into an
(N, N) advantage matrix downstream rather than erroring."""
import torch

REWARD_MAX = 8.0


def compute_reward(env, obs, action, info):
    d = torch.linalg.norm(env.agent.tcp.pose.p - env.cubeA.pose.p, axis=1)
    return (2 * (1 - torch.tanh(5 * d))).unsqueeze(1)
