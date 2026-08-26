import math

import torch

REWARD_MAX = 8.0

# ---------------------------------------------------------------------------
# Tunables (all metres). These encode the geometry of the failure mode:
# the hand descends into the narrow slot between the cubes and shoves cubeB.
# ---------------------------------------------------------------------------

# tcp height above cubeB's TOP FACE at which the hand/fingers are out of
# danger. Panda fingers are ~5 cm long above the tcp point, so 6 cm of
# clearance means nothing on the hand can reach cubeB.
_H_CLEAR = 0.06

# lateral offset of the tcp AWAY from cubeB (measured along the cubeA->cubeB
# horizontal axis, relative to cubeA's centre) that fully clears the
# intrusion penalty. 8 mm: enough to give the near finger room in a tight
# slot, small enough that cubeA is still inside the finger span.
_SIDE_HALF = 0.008

# cubeA-cubeB horizontal separation over which the intrusion penalty fades in.
# Below _OVER_LO cubeA is essentially over cubeB, i.e. the hand being above
# cubeB is the *intended* placement pose and must not be penalised.
_OVER_LO = 0.025
_OVER_HI = 0.045

# how much of the in-stage shaping the intrusion can remove.
_W_INTRUSION = 0.35


def compute_reward(env, obs, action, info):
    """Dense reward for StackCube-v1, biased towards clearing the neighbour cube.

    Ladder (bands are disjoint and ordered):
        stage A  approach, cubeA not grasped and not stacked : [0.0, 2.0]
        stage B  cubeA grasped, not yet stacked              : [3.0, 5.0]
        stage C  cubeA on cubeB, not yet success             : [6.0, 7.0]
        success                                             : 8.0
    """
    tcp = env.agent.tcp.pose.p                 # (N, 3)
    pA = env.cubeA.pose.p                      # (N, 3)  view into sim buffer
    pB = env.cubeB.pose.p                      # (N, 3)  view into sim buffer
    hs = env.cube_half_size[2]                 # 0.02

    # -----------------------------------------------------------------
    # Clearance term: "is the hand intruding into the slot next to cubeB?"
    #
    # Measured RELATIVE TO cubeA along the horizontal cubeA->cubeB axis, so
    # that shoving cubeB away cannot relieve the penalty (it only rotates the
    # axis). p_side > 0 means the tcp sits on the cubeB side of cubeA, i.e.
    # in the slot; p_side < 0 means it sits on the far side, which is the
    # behaviour we want during the descent and the lift.
    # -----------------------------------------------------------------
    ab_xy = pB[:, :2] - pA[:, :2]
    s_AB = torch.linalg.norm(ab_xy, dim=1)
    u_ab = ab_xy / torch.clamp(s_AB, min=1e-6).unsqueeze(-1)
    p_side = ((tcp[:, :2] - pA[:, :2]) * u_ab).sum(dim=1)
    side = torch.clamp((p_side + _SIDE_HALF) / (2.0 * _SIDE_HALF), 0.0, 1.0)

    # only dangerous while the hand is low enough to touch cubeB
    h_above = tcp[:, 2] - (pB[:, 2] + hs)
    height_gate = torch.clamp(1.0 - h_above / _H_CLEAR, 0.0, 1.0)

    # switched off once cubeA is (nearly) over cubeB: then the hand being
    # above cubeB is the placement itself, not an intrusion.
    not_over = torch.clamp((s_AB - _OVER_LO) / (_OVER_HI - _OVER_LO), 0.0, 1.0)

    intrusion = height_gate * side * not_over            # in [0, 1]
    clear = 1.0 - _W_INTRUSION * intrusion               # in [0.65, 1]

    # -----------------------------------------------------------------
    # Stage A: reach cubeA, and reach it from the side away from cubeB.
    # Two tanh scales so there is still gradient at the millimetre scale,
    # which is what a bounded residual can actually act on.
    # -----------------------------------------------------------------
    d_A = torch.linalg.norm(tcp - pA, dim=1)
    reach = 0.5 * (1.0 - torch.tanh(5.0 * d_A)) + 0.5 * (1.0 - torch.tanh(15.0 * d_A))
    reward = 2.0 * reach * clear                         # in [0, 2]

    # -----------------------------------------------------------------
    # Stage B: carry cubeA to the point one cube above cubeB (full 3D
    # distance), still discounted by intrusion so that lifting clear of the
    # slot before translating is worth more than dragging through it.
    # -----------------------------------------------------------------
    dxy = pA[:, :2] - pB[:, :2]
    dz = pA[:, 2] - pB[:, 2] - 2.0 * hs
    d_goal = torch.sqrt((dxy * dxy).sum(dim=1) + dz * dz + 1e-12)
    place = 0.5 * (1.0 - torch.tanh(5.0 * d_goal)) + 0.5 * (
        1.0 - torch.tanh(20.0 * d_goal)
    )
    stage_b = 3.0 + 2.0 * place * clear                  # in [3, 5]

    grasped = info["is_cubeA_grasped"].bool()
    on = info["is_cubeA_on_cubeB"].bool()
    success = info["success"].bool()

    reward = torch.where(grasped & (~on), stage_b, reward)

    # -----------------------------------------------------------------
    # Stage C: cubeA is on cubeB. Open the gripper and let it settle.
    # (The policy is already competent here; this mirrors the built-in.)
    # -----------------------------------------------------------------
    qpos = env.agent.robot.get_qpos()
    width = torch.clamp(env.agent.robot.get_qlimits()[:, -1, 1] * 2.0, min=1e-6)
    ungrasp = torch.clamp(qpos[:, -2:].sum(dim=1) / width, 0.0, 1.0)
    ungrasp = torch.where(grasped, ungrasp, torch.ones_like(ungrasp))

    v = torch.linalg.norm(env.cubeA.linear_velocity, dim=1)
    av = torch.linalg.norm(env.cubeA.angular_velocity, dim=1)
    static = torch.clamp(1.0 - torch.tanh(10.0 * v + av), 0.0, 1.0)

    stage_c = 6.0 + 0.5 * (ungrasp + static)             # in [6, 7]
    reward = torch.where(on, stage_c, reward)

    reward = torch.where(success, torch.full_like(reward, REWARD_MAX), reward)

    return torch.clamp(reward, 0.0, REWARD_MAX).to(dtype=torch.float32)
