"""Layer B: exceeds REWARD_MAX, so the normalised reward leaves [0, 1] and the
value function it feeds is scaled wrong."""
import torch

REWARD_MAX = 1.0


def compute_reward(env, obs, action, info):
    d = torch.linalg.norm(env.agent.tcp.pose.p - env.cubeA.pose.p, axis=1)
    return 50.0 * (1 - torch.tanh(d))
