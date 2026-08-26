import math

import torch

REWARD_MAX = 8.0


def _axis_alignment_with_world_z(q):
    """For a cube with quaternion q (wxyz), return max_i |<body_axis_i, world_z>|.

    A cube resting flat on any face gives 1.0. A cube balanced on an edge gives
    ~0.707, on a corner ~0.577. This is the "is it lying flat or is it rocking
    on an edge/corner" measure. Pure function of q.
    """
    q = q / (torch.linalg.norm(q, dim=1, keepdim=True) + 1e-9)
    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    # third row of the body->world rotation matrix == world z expressed in body frame
    c0 = torch.abs(2.0 * (x * z - w * y))
    c1 = torch.abs(2.0 * (y * z + w * x))
    c2 = torch.abs(1.0 - 2.0 * (x * x + y * y))
    return torch.maximum(torch.maximum(c0, c1), c2)


def compute_reward(env, obs, action, info):
    A = env.cubeA.pose.p                      # (n,3) red cube = goes on top
    B = env.cubeB.pose.p                      # (n,3) green cube = base of stack
    tcp = env.agent.tcp.pose.p                # (n,3)
    half = env.cube_half_size[2]              # 0.02

    # ---------------- geometric quantities ----------------
    goal_z = B[:, 2] + 2.0 * half
    goal = torch.stack([B[:, 0], B[:, 1], goal_z], dim=1)

    d_goal = torch.linalg.norm(A - goal, dim=1)                 # full 3D, all axes
    d_xy = torch.linalg.norm(A[:, :2] - B[:, :2], dim=1)        # lateral offset on the stack
    dz_err = torch.abs(A[:, 2] - goal_z)                        # seated height error
    d_tcp = torch.linalg.norm(tcp - A, dim=1)                   # gripper <-> red cube

    v = torch.linalg.norm(env.cubeA.linear_velocity, dim=1)
    av = torch.linalg.norm(env.cubeA.angular_velocity, dim=1)

    # ---------------- bounded [0,1] shaping primitives ----------------
    # goal proximity of the cube itself (3D, monotone in d_goal)
    g_goal = 1.0 - torch.tanh(4.0 * d_goal)
    # gripper proximity to the cube (only ever used multiplied by g_goal)
    f_tcp = 1.0 - torch.tanh(5.0 * d_tcp)
    # lateral centring on the support cube, sharp in the millimetre range
    centre = 1.0 - torch.tanh(40.0 * d_xy)
    # fully seated (not released from marginally too high)
    seated = 1.0 - torch.tanh(200.0 * dz_err)
    # residual motion of the cube
    settle = 1.0 - torch.tanh(10.0 * v + av)
    # lying flat vs rocking on an edge/corner
    upright = torch.clamp((_axis_alignment_with_world_z(env.cubeA.pose.q) - 0.85) / 0.15,
                          0.0, 1.0)
    # gripper openness, same measure the built-in reward uses
    qlim = env.agent.robot.get_qlimits()
    gripper_width = qlim[0, -1, 1] * 2.0
    openness = torch.clamp(
        torch.sum(env.agent.robot.get_qpos()[:, -2:], dim=1) / (gripper_width + 1e-9),
        0.0, 1.0,
    )
    # hand has withdrawn clear of the finished stack (so the retreat cannot brush it)
    clearance = torch.clamp((d_tcp - 0.035) / 0.06, 0.0, 1.0)

    # in-carry approach shaping: line up in xy first, only then get paid for
    # coming down to stack height (height term gated by lateral alignment)
    align = 1.0 - torch.tanh(5.0 * d_xy)
    height = 1.0 - torch.tanh(15.0 * dz_err)
    carry = 0.6 * align + 0.4 * align * height                  # in [0,1]

    # ---------------- stage masks ----------------
    grasped = info["is_cubeA_grasped"].bool()
    on = info["is_cubeA_on_cubeB"].bool()
    succ = info["success"].bool()

    m_carry = grasped & (~on)
    m_hold = on & grasped
    m_placed = on & (~grasped) & (~succ)
    m_done = succ

    # ---------------- ladder ----------------
    # Stage A [0, 2]: cube neither held nor stacked. Monotone decreasing in the
    # cube's distance to the goal for ANY gripper pose (the reach term is gated
    # by g_goal, so approaching the hand can never beat losing the stack).
    r = 2.0 * g_goal * (0.65 + 0.35 * f_tcp)

    # Stage B [2.5, 3.5]: cube is being carried.
    r_carry = 2.5 + 1.0 * carry

    # Stage C [4.0, 4.9]: cube is at the stack pose but still held. Shaping pushes
    # a clean, still, centred, fully seated release.
    r_hold = 4.0 + 0.9 * (0.45 * openness + 0.25 * settle + 0.15 * centre + 0.15 * seated)

    # Stage D [5.0, 6.5]: cube released on the stack, not yet settled.
    r_placed = 5.0 + 1.5 * (0.35 * centre + 0.35 * settle + 0.30 * upright)

    # Stage E [7.0, 8.0]: success. Extra paid only for *quality* of the finished
    # stack: deep centring margin, flat on its face, hand withdrawn clear.
    r_done = 7.0 + (0.45 * centre + 0.30 * upright + 0.25 * clearance)

    r = torch.where(m_carry, r_carry, r)
    r = torch.where(m_hold, r_hold, r)
    r = torch.where(m_placed, r_placed, r)
    r = torch.where(m_done, r_done, r)

    r = torch.clamp(r, 0.0, REWARD_MAX)
    return r.to(dtype=torch.float32)
