import math

import torch

REWARD_MAX = 8.0

# ---------------------------------------------------------------------------
# geometry constants, all metres.  Cubes are 40 mm (half size 0.02), so a cube
# presents a support radius between 0.020 (face-on) and 0.0283 (corner-on).
# ---------------------------------------------------------------------------
_FINGER_KEEPOUT = 0.045   # lateral distance from cubeB centre at which the
                          # gripper-clearance penalty starts
_FINGER_RAMP = 0.025      # penalty saturates 25 mm inside that -> 0.020 m
_GATE_ABOVE = 0.012       # free above cubeB's top face + 12 mm
_GATE_RAMP = 0.030        # gate ramps in over 30 mm of descent

_CUBEA_LAT_START = 0.062  # cubeA/cubeB centre distance at which contact
_CUBEA_LAT_RAMP = 0.022   # becomes possible; saturates at 0.040
_CUBEA_Z_RAMP = 0.020     # how far below cubeB's top face saturates

_PEN_W = 0.5              # how much of a stage's shaping a full penalty erases

# ladder bands
_S0_BASE, _S0_SPAN = 0.0, 2.0   # approach          -> [0.0, 2.0]
_S1_BASE, _S1_SPAN = 2.5, 2.0   # carry / place     -> [2.5, 4.5]
_S2_BASE, _S2_SPAN = 5.5, 1.0   # placed, not done  -> [5.5, 6.5]
_S3 = 8.0                       # success


def _tcp_y_axis(q):
    """Second column of the rotation matrix of a wxyz quaternion, batched."""
    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    ax = 2.0 * (x * y - w * z)
    ay = 1.0 - 2.0 * (x * x + z * z)
    az = 2.0 * (y * z + w * x)
    return torch.stack([ax, ay, az], dim=-1)


def compute_reward(env, obs, action, info):
    cubeA_p = env.cubeA.pose.p            # (b, 3)
    cubeB_p = env.cubeB.pose.p            # (b, 3)
    tcp_p = env.agent.tcp.pose.p          # (b, 3)
    tcp_q = env.agent.tcp.pose.q          # (b, 4) wxyz
    qpos = env.agent.robot.get_qpos()     # (b, 9)

    hz = env.cube_half_size[2]            # 0.02

    # ------------------------------------------------------------------
    # base shaping quantities
    # ------------------------------------------------------------------
    goal = torch.stack(
        [cubeB_p[:, 0], cubeB_p[:, 1], cubeB_p[:, 2] + 2.0 * hz], dim=-1
    )
    d_goal = torch.linalg.norm(cubeA_p - goal, dim=1)          # full 3-D
    place_shape = 1.0 - torch.tanh(5.0 * d_goal)

    d_reach = torch.linalg.norm(tcp_p - cubeA_p, dim=1)        # full 3-D
    reach_shape = 1.0 - torch.tanh(5.0 * d_reach)

    # ------------------------------------------------------------------
    # gripper clearance from cubeB.  Three colliding points: the tcp origin
    # (palm/hand centre) and the two fingertips, offset along the tcp frame's
    # y axis by the two finger joint values.
    # ------------------------------------------------------------------
    y_axis = _tcp_y_axis(tcp_q)
    q_l = torch.clamp(qpos[:, -2], min=0.0).unsqueeze(-1)
    q_r = torch.clamp(qpos[:, -1], min=0.0).unsqueeze(-1)
    f_l = tcp_p + y_axis * q_l
    f_r = tcp_p - y_axis * q_r
    pts = torch.stack([tcp_p, f_l, f_r], dim=1)                # (b, 3, 3)

    lat = torch.linalg.norm(pts[:, :, :2] - cubeB_p[:, None, :2], dim=-1)
    lat_pen = torch.clamp((_FINGER_KEEPOUT - lat) / _FINGER_RAMP, 0.0, 1.0)

    z_top = (cubeB_p[:, 2] + hz).unsqueeze(-1)                 # cubeB top face
    z_gate = torch.clamp(
        (z_top + _GATE_ABOVE - pts[:, :, 2]) / _GATE_RAMP, 0.0, 1.0
    )
    pen_grip = torch.amax(lat_pen * z_gate, dim=1)             # (b,)

    # ------------------------------------------------------------------
    # carried-cube clearance: cubeA laterally close enough to touch cubeB
    # while it is still below cubeB's top face.  Exactly zero at the goal.
    # ------------------------------------------------------------------
    lat_a = torch.linalg.norm(cubeA_p[:, :2] - cubeB_p[:, :2], dim=1)
    ov_a = torch.clamp((_CUBEA_LAT_START - lat_a) / _CUBEA_LAT_RAMP, 0.0, 1.0)
    vz_a = torch.clamp((goal[:, 2] - cubeA_p[:, 2]) / _CUBEA_Z_RAMP, 0.0, 1.0)
    pen_cubeA = ov_a * vz_a

    pen_carry = torch.maximum(pen_grip, pen_cubeA)

    # ------------------------------------------------------------------
    # release / settle shaping (kept from the built-in: settling is fine)
    # ------------------------------------------------------------------
    grasped = info["is_cubeA_grasped"]
    on_B = info["is_cubeA_on_cubeB"]
    success = info["success"]

    width = torch.clamp(qpos[:, -2], min=0.0) + torch.clamp(qpos[:, -1], min=0.0)
    qlim = env.agent.robot.get_qlimits()
    gmax = torch.clamp(qlim[..., -1, 1] * 2.0, min=1e-4)
    ungrasp = torch.clamp(width / gmax, 0.0, 1.0)
    ungrasp = torch.where(grasped, ungrasp, torch.ones_like(ungrasp))

    v = torch.linalg.norm(env.cubeA.linear_velocity, dim=1)
    av = torch.linalg.norm(env.cubeA.angular_velocity, dim=1)
    static = 1.0 - torch.tanh(10.0 * v + av)

    # ------------------------------------------------------------------
    # the ladder
    # ------------------------------------------------------------------
    shape0 = 0.8 * reach_shape + 0.2 * place_shape
    r0 = _S0_BASE + _S0_SPAN * torch.clamp(shape0 - _PEN_W * pen_grip, 0.0, 1.0)

    r1 = _S1_BASE + _S1_SPAN * torch.clamp(
        place_shape - _PEN_W * pen_carry, 0.0, 1.0
    )

    r2 = _S2_BASE + _S2_SPAN * (0.5 * ungrasp + 0.5 * static)

    reward = r0
    reward = torch.where(grasped, r1, reward)
    reward = torch.where(on_B, r2, reward)
    reward = torch.where(success, torch.full_like(reward, _S3), reward)

    reward = torch.clamp(reward, 0.0, REWARD_MAX)
    return reward.to(torch.float32)
