"""Layer A: wraps the reward in a class. Harmless in itself, but it means the
module-level function the loader looks for does not exist."""
import torch

REWARD_MAX = 8.0


class Reward:
    def compute_reward(self, env, obs, action, info):
        return torch.zeros(env.num_envs)
