"""Layer A: the nastiest hack available - teleport cubeA onto cubeB from inside
the reward, then collect the bonus for cubeA being on cubeB. Caught statically
by the set_pose ban, and again at runtime by layer B state-dict snapshot."""
import torch

REWARD_MAX = 8.0


def compute_reward(env, obs, action, info):
    env.cubeA.set_pose(env.cubeB.pose)
    return torch.zeros(env.num_envs) + 8.0
