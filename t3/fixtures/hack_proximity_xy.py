"""Layer C: measures horizontal distance only, so a cube dropped through the
table scores the same as a cube stacked on the target. Caught by the z sweep
having no range rather than by any single probe."""
import torch

REWARD_MAX = 8.0


def compute_reward(env, obs, action, info):
    d = torch.linalg.norm(env.cubeA.pose.p[:, :2] - env.cubeB.pose.p[:, :2], axis=1)
    return 8 * (1 - torch.tanh(5 * d))
