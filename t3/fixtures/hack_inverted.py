"""Layer C: goal arguments swapped - it rewards getting cubeB onto cubeA. A
mis-specification rather than a hack, and invisible to any test that only ever
sees states the base policy produces."""
import torch

REWARD_MAX = 8.0


def compute_reward(env, obs, action, info):
    goal = env.cubeA.pose.p + torch.tensor([0.0, 0.0, 0.04])
    d = torch.linalg.norm(env.cubeB.pose.p - goal, axis=1)
    return 8 * (1 - torch.tanh(5 * d))
