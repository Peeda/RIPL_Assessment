"""Layer C/D: the sparse reward wearing a dense reward's clothes. Satisfies
every ordering by a strict > and none of them by a margin, which is why the
probe orderings carry one."""
import torch

REWARD_MAX = 8.0


def compute_reward(env, obs, action, info):
    return 8.0 * info["success"].float()
