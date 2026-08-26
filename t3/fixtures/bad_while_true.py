"""Layer A: an unbounded loop. In a sampler this hangs the environment reset;
in a reward it hangs every step of training."""
import torch

REWARD_MAX = 8.0


def compute_reward(env, obs, action, info):
    reward = torch.zeros(env.num_envs)
    i = 0
    while True:
        i = i + 1
        if i > 10:
            break
    return reward
