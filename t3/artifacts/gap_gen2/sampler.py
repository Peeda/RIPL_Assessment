import math

import torch

# ---------------------------------------------------------------------------
# Biased episode sampler for the "tight face clearance" (gap) failure region.
#
# Quantity being biased: the FACE gap between the two cubes measured along the
# line joining their centres,
#       gap = d - s_A - s_B,
#       s = 0.02 * (|cos(psi)| + |sin(psi)|)   in [0.020, 0.028284],
# where psi is the angle between that line and the cube's own x-axis (mod 90
# deg). s is the cube's half-extent along the centre line: 0.020 face-first,
# 0.028284 corner-first. The environment rejects d < 0.05857, so the tightest
# physically reachable gap is 0.05857 - 2*0.028284 = 0.002 m, and it is only
# reachable when BOTH cubes are turned corner-first towards each other.
#
# This sampler draws that configuration directly: corner-first yaws (psi within
# 18 deg of 45 deg) and a gap drawn from ~[0.002, 0.012] with a small tail out
# to 0.024. Direction of the pair, absolute yaw (mod 90 deg has 4 branches) and
# the pair's location on the table are all fully varied.
# ---------------------------------------------------------------------------

_Z = 0.02
_HS = 0.02
_MIN_SEP = 0.05860          # env floor 0.05857 + 3e-5 safety
_X_LIM = 0.2
_Y_LIM = 0.3
_BASE_X = -0.615
_BASE_Y = 0.0
_REACH = 0.868
_PSI_SPREAD = 18.0 * math.pi / 180.0
_WIDE_FRAC = 0.12           # fraction of rows given a slightly easier gap


def _draw(b, device, clamp_mid):
    """One vectorised candidate draw. Returns (ax, ay, bx, by, th_a, th_b)."""
    # direction from cubeB towards cubeA
    alpha = torch.rand(b, device=device) * (2.0 * math.pi)

    # yaw of each cube relative to the centre line: near 45 deg == corner-first
    psi_a = math.pi / 4.0 + (torch.rand(b, device=device) - 0.5) * 2.0 * _PSI_SPREAD
    psi_b = math.pi / 4.0 + (torch.rand(b, device=device) - 0.5) * 2.0 * _PSI_SPREAD

    s_a = _HS * (torch.abs(torch.cos(psi_a)) + torch.abs(torch.sin(psi_a)))
    s_b = _HS * (torch.abs(torch.cos(psi_b)) + torch.abs(torch.sin(psi_b)))

    # target face gap: tight core plus a small tail for variety
    gap = 0.0005 + torch.rand(b, device=device) * 0.0115
    wide = (torch.rand(b, device=device) < _WIDE_FRAC).to(gap.dtype)
    gap = gap + wide * (0.004 + torch.rand(b, device=device) * 0.008)

    # centre-to-centre distance, never below the environment's own floor
    d = torch.clamp(s_a + s_b + gap, min=_MIN_SEP)

    ux = torch.cos(alpha)
    uy = torch.sin(alpha)

    # midpoint of the pair, mimicking the nominal single-cube marginals
    # (x: triangular on [-0.2, 0.2]; y: trapezoid on [-0.3, 0.3])
    mx = 0.2 * (torch.rand(b, device=device) - 0.5) + 0.2 * (
        torch.rand(b, device=device) - 0.5
    )
    my = 0.2 * (torch.rand(b, device=device) - 0.5) + 0.4 * (
        torch.rand(b, device=device) - 0.5
    )
    if clamp_mid:
        # guaranteed-valid fallback: half a separation (<=0.035) inside the box
        mx = torch.clamp(mx, -0.15, 0.15)
        my = torch.clamp(my, -0.24, 0.24)

    half = 0.5 * d
    ax = mx + half * ux
    ay = my + half * uy
    bx = mx - half * ux
    by = my - half * uy

    # absolute yaws. The cube is 4-fold symmetric about z, so adding k*90 deg
    # leaves the geometry (and s) untouched while covering the full yaw range
    # the base policy was trained on.
    ka = torch.randint(0, 4, (b,), device=device).to(alpha.dtype)
    kb = torch.randint(0, 4, (b,), device=device).to(alpha.dtype)
    th_a = alpha - psi_a + ka * (math.pi / 2.0)
    th_b = alpha - psi_b + kb * (math.pi / 2.0)

    return ax, ay, bx, by, th_a, th_b


def _valid(ax, ay, bx, by):
    inside = (
        (ax.abs() <= _X_LIM)
        & (bx.abs() <= _X_LIM)
        & (ay.abs() <= _Y_LIM)
        & (by.abs() <= _Y_LIM)
    )
    ra = torch.sqrt((ax - _BASE_X) ** 2 + (ay - _BASE_Y) ** 2)
    rb = torch.sqrt((bx - _BASE_X) ** 2 + (by - _BASE_Y) ** 2)
    return inside & (ra <= _REACH) & (rb <= _REACH)


def sample_cube_poses(b, device):
    # start from a draw that is valid by construction
    ax, ay, bx, by, th_a, th_b = _draw(b, device, True)
    accepted = _valid(ax, ay, bx, by)

    for _ in range(48):
        cax, cay, cbx, cby, cta, ctb = _draw(b, device, False)
        ok = _valid(cax, cay, cbx, cby)
        upd = ok & (~accepted)
        ax = torch.where(upd, cax, ax)
        ay = torch.where(upd, cay, ay)
        bx = torch.where(upd, cbx, bx)
        by = torch.where(upd, cby, by)
        th_a = torch.where(upd, cta, th_a)
        th_b = torch.where(upd, ctb, th_b)
        accepted = accepted | ok

    # final hard guarantee (no-op for rows already inside the support)
    ax = torch.clamp(ax, -_X_LIM, _X_LIM)
    bx = torch.clamp(bx, -_X_LIM, _X_LIM)
    ay = torch.clamp(ay, -_Y_LIM, _Y_LIM)
    by = torch.clamp(by, -_Y_LIM, _Y_LIM)

    z = torch.full((b,), _Z, device=device, dtype=torch.float32)
    cubeA_xyz = torch.stack([ax, ay, z], dim=1).to(dtype=torch.float32)
    cubeB_xyz = torch.stack([bx, by, z], dim=1).to(dtype=torch.float32)

    zero = torch.zeros(b, device=device, dtype=torch.float32)
    ha = 0.5 * th_a
    hb = 0.5 * th_b
    cubeA_quat = torch.stack(
        [torch.cos(ha), zero, zero, torch.sin(ha)], dim=1
    ).to(dtype=torch.float32)
    cubeB_quat = torch.stack(
        [torch.cos(hb), zero, zero, torch.sin(hb)], dim=1
    ).to(dtype=torch.float32)

    return {
        "cubeA_xyz": cubeA_xyz,
        "cubeA_quat": cubeA_quat,
        "cubeB_xyz": cubeB_xyz,
        "cubeB_quat": cubeB_quat,
    }
