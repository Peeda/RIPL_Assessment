import math

import torch

# Panda base (see _load_agent: sapien.Pose(p=[-0.615, 0, 0]))
_BASE_X = -0.615
_BASE_Y = 0.0

# Nominal support of the environment's own sampler, with a small safety margin.
# Both cubes live in x in [-0.2, 0.2], y in [-0.3, 0.3]; additionally the shared
# xy offset in _initialize_episode means the pair's relative displacement is
# bounded by |dx| <= 0.2, |dy| <= 0.4. We stay strictly inside all of that.
_X_LIM = 0.195
_Y_LIM = 0.295
_DX_LIM = 0.19
_DY_LIM = 0.37

# Failure region: the TARGET (green, cubeB) sits on the far arc of the
# comfortably reachable workspace; the red cube starts comfortable and clear.
_RB_MIN, _RB_MAX = 0.695, 0.775     # cubeB radius from the base
_TB_MAX = 0.36                      # cubeB bearing spread (rad), ~+-20.6 deg
_RA_MIN, _RA_MAX = 0.500, 0.685     # cubeA radius from the base
_TA_MAX = 0.50                      # cubeA bearing spread (rad), ~+-28.6 deg
_MIN_SEP = 0.08                     # >> 0.05857 floor: "well clear of each other"

_ATTEMPTS = 48


def _yaw_quat(angle):
    half = 0.5 * angle
    w = torch.cos(half)
    z = torch.sin(half)
    zeros = torch.zeros_like(w)
    q = torch.stack([w, zeros, zeros, z], dim=1)
    return q / (torch.linalg.norm(q, dim=1, keepdim=True) + 1e-9)


def sample_cube_poses(b, device):
    # Guaranteed-valid fallback (checked by hand against every constraint):
    # B = (0.140, 0.050) -> r = 0.757 ; A = (-0.020, -0.050) -> r = 0.597
    # |dx| = 0.16, |dy| = 0.10, sep = 0.189
    A_xy = torch.zeros((b, 2), dtype=torch.float32)
    B_xy = torch.zeros((b, 2), dtype=torch.float32)
    A_xy[:, 0] = -0.020
    A_xy[:, 1] = -0.050
    B_xy[:, 0] = 0.140
    B_xy[:, 1] = 0.050

    filled = torch.zeros((b,), dtype=torch.bool)

    for _ in range(_ATTEMPTS):
        rB = _RB_MIN + torch.rand((b,)) * (_RB_MAX - _RB_MIN)
        tB = (torch.rand((b,)) * 2.0 - 1.0) * _TB_MAX
        rA = _RA_MIN + torch.rand((b,)) * (_RA_MAX - _RA_MIN)
        tA = (torch.rand((b,)) * 2.0 - 1.0) * _TA_MAX

        Bx = _BASE_X + rB * torch.cos(tB)
        By = _BASE_Y + rB * torch.sin(tB)
        Ax = _BASE_X + rA * torch.cos(tA)
        Ay = _BASE_Y + rA * torch.sin(tA)

        dx = Ax - Bx
        dy = Ay - By
        sep = torch.sqrt(dx * dx + dy * dy + 1e-12)

        ok = (torch.abs(Bx) <= _X_LIM) & (torch.abs(By) <= _Y_LIM)
        ok = ok & (torch.abs(Ax) <= _X_LIM) & (torch.abs(Ay) <= _Y_LIM)
        ok = ok & (torch.abs(dx) <= _DX_LIM) & (torch.abs(dy) <= _DY_LIM)
        ok = ok & (sep >= _MIN_SEP)
        # both cubes comfortably inside the 0.8 m IK ceiling
        ok = ok & (rB <= 0.79) & (rA <= 0.79)

        new = ok & (~filled)
        cand_A = torch.stack([Ax, Ay], dim=1)
        cand_B = torch.stack([Bx, By], dim=1)
        A_xy[new] = cand_A[new]
        B_xy[new] = cand_B[new]
        filled = filled | ok

    cubeA_xyz = torch.zeros((b, 3), dtype=torch.float32)
    cubeB_xyz = torch.zeros((b, 3), dtype=torch.float32)
    cubeA_xyz[:, :2] = A_xy
    cubeB_xyz[:, :2] = B_xy
    cubeA_xyz[:, 2] = 0.02
    cubeB_xyz[:, 2] = 0.02

    # yaw left fully uniform, exactly as the environment does it: the frozen
    # policy must still cope with arbitrary cube yaw in this region.
    yawA = (torch.rand((b,)) * 2.0 - 1.0) * math.pi
    yawB = (torch.rand((b,)) * 2.0 - 1.0) * math.pi

    return {
        "cubeA_xyz": cubeA_xyz,
        "cubeA_quat": _yaw_quat(yawA),
        "cubeB_xyz": cubeB_xyz,
        "cubeB_quat": _yaw_quat(yawB),
    }
