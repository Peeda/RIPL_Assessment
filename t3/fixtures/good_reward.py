"""A hand-written reward that satisfies the contract and passes every layer.

Not a mistake fixture: this is the positive control. A checker that has only
ever been shown bad inputs is one nobody has confirmed can accept anything.

It is also the shape a good generation should have, so it doubles as the
worked example the prompt points at. Structure: the stock 8-stage ladder
(reach -> carry -> settle -> release), plus one term aimed at mode `gap`'s
measured mechanism - while carrying cubeA, the gripper is penalised for being
close to cubeB, because T-II found that mode A's failures are descents that
foul the neighbouring cube.

Note what the clearance term CANNOT be. The reward is stateless by contract, so
it cannot compare cubeB's pose against where cubeB started and penalise having
shoved it. It can only shape the behaviour that causes the shove. That
limitation is worth knowing before reading a generation that tries anyway.
"""
import torch

REWARD_MAX = 8.0


def compute_reward(env, obs, action, info):
    tcp_pos = env.agent.tcp.pose.p
    cubeA_pos = env.cubeA.pose.p
    cubeB_pos = env.cubeB.pose.p

    grasped = info["is_cubeA_grasped"]
    on_b = info["is_cubeA_on_cubeB"]

    # --- stage 1: reach cubeA ---------------------------------------------
    reach = torch.linalg.norm(tcp_pos - cubeA_pos, axis=1)
    reward = 2 * (1 - torch.tanh(5 * reach))

    # --- stage 2: carry it over cubeB -------------------------------------
    goal = torch.hstack(
        [cubeB_pos[:, 0:2], (cubeB_pos[:, 2] + env.cube_half_size[2] * 2)[:, None]]
    )
    place = 1 - torch.tanh(5.0 * torch.linalg.norm(goal - cubeA_pos, axis=1))

    # The mode-`gap` term. While cubeA is in the gripper, the fingers should
    # stay clear of cubeB in the horizontal plane: a descent that comes down on
    # top of the neighbour is what displaces it, and a displaced cubeB is what
    # T-II measured behind mode A's placement collapse (29.3 mm in failures
    # against 3.0 mm in successes). Bounded at 0.5 so it can shade the carry
    # stage without ever reordering the ladder.
    tcp_to_b_xy = torch.linalg.norm(tcp_pos[:, :2] - cubeB_pos[:, :2], axis=1)
    clearance = 0.5 * torch.tanh(20.0 * tcp_to_b_xy)
    reward[grasped] = (3.5 + place + clearance)[grasped]

    # --- stage 3: let go, and let it settle --------------------------------
    gripper_width = env.agent.robot.get_qlimits()[0, -1, 1] * 2
    opened = torch.sum(env.agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
    opened[~grasped] = 1.0
    v = torch.linalg.norm(env.cubeA.linear_velocity, axis=1)
    av = torch.linalg.norm(env.cubeA.angular_velocity, axis=1)
    settled = 1 - torch.tanh(v * 10 + av)
    reward[on_b] = (6 + (opened + settled) / 2.0)[on_b]

    reward[info["success"]] = REWARD_MAX
    return torch.clamp(reward, 0.0, REWARD_MAX)
