import math

import torch

# Panda base (see _load_agent)
BASE_X = -0.615
BASE_Y = 0.0

# nominal support of the environment's own sampler
X_MIN = -0.2
X_MAX = 0.2
Y_ABS = 0.3

# failure region: only the TARGET cube is far
D_B_MIN = 0.78    # far tail of the base->cube distance distribution (~2.7% of draws)
D_B_MAX = 0.866   # the environment's own far corner, minus a hair for the 0.868 check
D_A_MAX = 0.70    # cubeA stays comfortable

SEP_MIN = 0.09    # "well clear of each other" (env's physical floor is 0.05857)

# joint-support constraints implied by the env's single shared xy offset:
# both cubes must fit in one translated placement box -> |dx| <= 0.2, |dy| <= 0.4
DX_MAX = 0.195
DY_MAX = 0.39


def sample_cube_poses(b, device):
    # ------------------------------------------------------------------ defaults
    # Guaranteed-valid, still varied.  Used as the fall-back if a rejection
    # budget runs out (probability ~1e-8 per row).
    bx = 0.185 + 0.015 * torch.rand(b)                 # d_B in [0.800, 0.850]
    by = (torch.rand(b) * 2.0 - 1.0) * 0.24
    ax = 0.005 + 0.040 * torch.rand(b)                 # d_A <= 0.671
    ay = (torch.rand(b) * 2.0 - 1.0) * 0.14

    # ------------------------------------------------------------------ cubeB: far target
    # Uniform over {x in [0.105, 0.2], |y| <= 0.3, D_B_MIN <= d <= D_B_MAX}.
    # Acceptance of the proposal box is ~58%.
    valid_b = torch.zeros(b, dtype=torch.bool)
    for _ in range(24):
        cx = 0.105 + (X_MAX - 0.105) * torch.rand(b)
        cy = (torch.rand(b) * 2.0 - 1.0) * Y_ABS
        d = torch.sqrt((cx - BASE_X) ** 2 + (cy - BASE_Y) ** 2)
        ok = (d >= D_B_MIN) & (d <= D_B_MAX)
        take = ok & (~valid_b)
        bx = torch.where(take, cx, bx)
        by = torch.where(take, cy, by)
        valid_b = valid_b | ok

    # ------------------------------------------------------------------ cubeA: comfortable
    valid_a = torch.zeros(b, dtype=torch.bool)
    lo_x = torch.clamp(bx - DX_MAX, min=X_MIN)
    span_x = torch.clamp(X_MAX - lo_x, min=1e-4)
    for _ in range(40):
        cx = lo_x + span_x * torch.rand(b)
        cy = (torch.rand(b) * 2.0 - 1.0) * Y_ABS
        d = torch.sqrt((cx - BASE_X) ** 2 + (cy - BASE_Y) ** 2)
        sep = torch.sqrt((cx - bx) ** 2 + (cy - by) ** 2)
        ok = (
            (d <= D_A_MAX)
            & (sep >= SEP_MIN)
            & (torch.abs(cx - bx) <= DX_MAX)
            & (torch.abs(cy - by) <= DY_MAX)
        )
        take = ok & (~valid_a)
        ax = torch.where(take, cx, ax)
        ay = torch.where(take, cy, ay)
        valid_a = valid_a | ok

    # ------------------------------------------------------------------ final clamps
    ax = torch.clamp(ax, X_MIN, X_MAX)
    bx = torch.clamp(bx, X_MIN, X_MAX)
    ay = torch.clamp(ay, -Y_ABS, Y_ABS)
    by = torch.clamp(by, -Y_ABS, Y_ABS)

    z = torch.full((b,), 0.02, dtype=torch.float32)
    cubeA_xyz = torch.stack([ax, ay, z], dim=-1).to(torch.float32)
    cubeB_xyz = torch.stack([bx, by, z], dim=-1).to(torch.float32)

    # ------------------------------------------------------------------ yaw only
    zeros = torch.zeros(b, dtype=torch.float32)

    tha = torch.rand(b) * (2.0 * math.pi)
    cubeA_quat = torch.stack(
        [torch.cos(tha * 0.5), zeros, zeros, torch.sin(tha * 0.5)], dim=-1
    ).to(torch.float32)

    thb = torch.rand(b) * (2.0 * math.pi)
    cubeB_quat = torch.stack(
        [torch.cos(thb * 0.5), zeros, zeros, torch.sin(thb * 0.5)], dim=-1
    ).to(torch.float32)

    return {
        "cubeA_xyz": cubeA_xyz,
        "cubeA_quat": cubeA_quat,
        "cubeB_xyz": cubeB_xyz,
        "cubeB_quat": cubeB_quat,
    }
