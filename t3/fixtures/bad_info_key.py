"""Layer A: reads a flag evaluate() does not return. A KeyError at step 40,000
of a training run, for a typo that costs 0.2 s to catch here."""
import torch

REWARD_MAX = 8.0


def compute_reward(env, obs, action, info):
    return torch.zeros(env.num_envs) + info["is_cube_a_grasped"].float()
