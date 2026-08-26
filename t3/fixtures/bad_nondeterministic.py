"""Layer B: draws randomness inside the reward, so the same state scores
differently on two consecutive calls and the advantage estimate carries noise
that no amount of training averages out."""
import torch

REWARD_MAX = 8.0


def compute_reward(env, obs, action, info):
    d = torch.linalg.norm(env.agent.tcp.pose.p - env.cubeA.pose.p, axis=1)
    base = 2 * (1 - torch.tanh(5 * d))
    return torch.clamp(base + 0.1 * torch.rand_like(base), 0.0, REWARD_MAX)
