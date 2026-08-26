import math

import torch

REWARD_MAX = 8.0


def compute_reward(env, obs, action, info):
    """Dense reward for StackCube-v1, shaped for the 'target cube at the edge of
    the workspace' failure mode: the stack is *made* but does not *stay*.

    Ladder (strictly ordered bands, see rationale):
        [0.0, 1.5]  approach cubeA           (not grasped, not placed)
        [3.0, 4.8]  carry cubeA to the goal  (grasped, not placed)
        [5.0, 6.2]  placed but still held    (on cubeB, grasped)
        [6.4, 7.8]  placed and released      (on cubeB, not grasped, not static)
         8.0        success                  (on cubeB, static, released)
    """
    pA = env.cubeA.pose.p
    pB = env.cubeB.pose.p
    tcp = env.agent.tcp.pose.p

    half = env.cube_half_size[2]
    cube_size = 2.0 * half  # 0.04 m, the stacked z offset

    # ------------------------------------------------------------------ geometry
    d_tcp_A = torch.linalg.norm(tcp - pA, dim=-1)

    goal = torch.stack(
        [pB[:, 0], pB[:, 1], pB[:, 2] + cube_size], dim=-1
    )  # cubeA must end up here: directly ABOVE cubeB
    d_goal = torch.linalg.norm(pA - goal, dim=-1)

    xy_err = torch.linalg.norm(pA[:, :2] - pB[:, :2], dim=-1)
    z_err = torch.abs((pA[:, 2] - pB[:, 2]) - cube_size)

    # ------------------------------------------------------------------ motion of cubeA
    v = torch.linalg.norm(env.cubeA.linear_velocity, dim=-1)
    av = torch.linalg.norm(env.cubeA.angular_velocity, dim=-1)
    # calibrated against evaluate()'s static thresholds (lin 1e-2, ang 0.5)
    low_motion = 1.0 - torch.tanh(10.0 * v + av)

    # ------------------------------------------------------------------ cubeA flatness
    # World-z row of cubeA's rotation matrix.  A cube is "settled flat" if ANY of
    # its body axes is vertical (six equivalent face-down orientations), so take
    # the largest |component|.  This is 1 when a face lies flat, and drops when
    # the cube is perched on an edge or a corner and rocking.
    q = env.cubeA.pose.q
    qw = q[:, 0]
    qx = q[:, 1]
    qy = q[:, 2]
    qz = q[:, 3]
    r20 = 2.0 * (qx * qz - qw * qy)
    r21 = 2.0 * (qy * qz + qw * qx)
    r22 = 1.0 - 2.0 * (qx * qx + qy * qy)
    flat = torch.maximum(torch.maximum(torch.abs(r20), torch.abs(r21)), torch.abs(r22))
    upright = torch.clamp((flat - 0.8) / 0.2, 0.0, 1.0)

    # ------------------------------------------------------------------ gripper opening
    qpos = env.agent.robot.get_qpos()
    qlim = env.agent.robot.get_qlimits()
    width = qlim[0, -1, 1] * 2.0  # full open finger-joint sum (Panda)
    open_frac = torch.clamp(
        torch.sum(qpos[:, -2:], dim=-1) / (torch.abs(width) + 1e-6), 0.0, 1.0
    )

    # ------------------------------------------------------------------ placement quality
    # Tight scales: this is the resolution the built-in reward does not have once
    # is_cubeA_on_cubeB flips true.  Off-centre / perched / tilted placements are
    # what topple at full arm extension.
    xy_precision = 1.0 - torch.tanh(xy_err / 0.012)
    z_precision = 1.0 - torch.tanh(z_err / 0.004)
    place_quality = 0.50 * xy_precision + 0.20 * z_precision + 0.30 * upright
    place_quality = torch.clamp(place_quality, 0.0, 1.0)

    # vertical clearance of the tool from the top face of cubeA: shapes a clean
    # upward withdrawal instead of a sideways one that brushes the fresh stack.
    clear = torch.clamp((tcp[:, 2] - (pA[:, 2] + half)) / 0.05, 0.0, 1.0)

    # ------------------------------------------------------------------ stage values
    # Stage A: approach (bounded well below the grasp band).
    r_approach = 1.5 * (1.0 - torch.tanh(5.0 * d_tcp_A))

    # Stage B: carry to the goal, with a small "calm when close" term so the cube
    # arrives over cubeB without residual sideways speed.
    near_goal = 1.0 - torch.tanh(20.0 * d_goal)
    r_carry = (
        3.0
        + 1.5 * (1.0 - torch.tanh(5.0 * d_goal))
        + 0.3 * near_goal * low_motion
    )

    # Stage C: cube is nominally stacked but still in the fingers.  Pays for
    # squareness of the placement and for opening the hand.  Max 6.2.
    r_held = 5.0 + 0.8 * place_quality + 0.4 * open_frac

    # Stage D: cube is stacked and let go, but not yet certified static.  Max 7.8.
    r_released = 6.4 + 1.0 * place_quality + 0.3 * low_motion + 0.1 * clear

    # ------------------------------------------------------------------ assemble
    grasped = info["is_cubeA_grasped"].bool()
    placed = info["is_cubeA_on_cubeB"].bool()
    success = info["success"].bool()

    reward = r_approach
    reward = torch.where(grasped & (~placed), r_carry, reward)
    reward = torch.where(placed & grasped, r_held, reward)
    reward = torch.where(placed & (~grasped), r_released, reward)
    reward = torch.where(success, torch.full_like(reward, REWARD_MAX), reward)

    reward = torch.clamp(reward, 0.0, REWARD_MAX)
    return reward.to(dtype=torch.float32)
