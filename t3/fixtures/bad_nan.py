"""Layer B: divides by a distance that is zero exactly when the gripper reaches
the cube - i.e. at the moment the reward matters most."""
import torch

REWARD_MAX = 8.0


def compute_reward(env, obs, action, info):
    d = torch.linalg.norm(env.agent.tcp.pose.p - env.cubeA.pose.p, axis=1)
    return torch.clamp(1.0 / d, 0.0, REWARD_MAX)
