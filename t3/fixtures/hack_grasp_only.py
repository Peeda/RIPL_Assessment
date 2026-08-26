"""Layer C/D: the commonest reward hack. Pays richly for reaching and grasping
cubeA and never mentions cubeB, so "pick it up and hold it forever" is optimal.
Scores a high UNCONDITIONAL alignment AUC, because grasping correlates with
everything downstream - and lands at chance on the stage-conditional test. That
is exactly why layer D conditions on the stage the mode fails at."""
import torch

REWARD_MAX = 8.0


def compute_reward(env, obs, action, info):
    d = torch.linalg.norm(env.agent.tcp.pose.p - env.cubeA.pose.p, axis=1)
    reward = 4 * (1 - torch.tanh(5 * d))
    reward[info["is_cubeA_grasped"]] = 8.0
    return reward
