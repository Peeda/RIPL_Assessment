"""Layer B: returns a Python float instead of a per-environment tensor. Works at
num_envs=1 by broadcasting, silently wrong at every other width."""
import torch

REWARD_MAX = 8.0


def compute_reward(env, obs, action, info):
    d = torch.linalg.norm(env.agent.tcp.pose.p - env.cubeA.pose.p, axis=1)
    return float(d.mean())
