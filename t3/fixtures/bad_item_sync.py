"""Layer A: .item() forces a GPU->CPU sync on every step. Invisible at the
num_envs=1 width a casual test uses, and it dominates wall clock at the batch
width PPO actually trains at."""
import torch

REWARD_MAX = 8.0


def compute_reward(env, obs, action, info):
    d = torch.linalg.norm(env.agent.tcp.pose.p - env.cubeA.pose.p, axis=1)
    scale = d.mean().item()
    return torch.zeros(env.num_envs) + scale
