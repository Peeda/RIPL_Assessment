import math

import torch


def sample_cube_poses(b, device):
    """Initial cube configurations biased toward the 'tight face clearance' regime.

    The failure region is a statement about *face* gap, not centre distance:
    each 40 mm cube presents a half-extent of 0.020 m face-on and 0.0283 m
    corner-on along a given direction.  We therefore build each configuration
    from the quantity that actually matters:

        gap = d_centres - sup_A(u) - sup_B(u)

    where u is the unit vector between the two centres and
    sup(u) = s * (|cos delta| + |sin delta|), delta = angle between u and the
    cube's own x axis.  We sample the yaws (mostly corner-first, i.e.
    delta ~ 45 deg) and a small target gap, then solve for the centre distance,
    finally clamping to the environment's own rejection floor (0.05857 m).

    Everything else -- the bearing of the pair, where the pair sits on the
    table -- is left broadly random so the residual cannot overfit one corner
    of the region.
    """
    s = 0.02
    MIN_SEP = 0.05857
    SEP_EPS = 2.0e-4

    # pair-centre region: a strict subset of the nominal support, chosen so that
    # every cube lands inside x in [-0.2, 0.2], y in [-0.3, 0.3] and within
    # 0.8 m of the Panda base at (-0.615, 0.0).
    X_LO, X_HI = -0.12, 0.09
    Y_LO, Y_HI = -0.15, 0.15
    BASE_X, BASE_Y = -0.615, 0.0
    MAX_BASE_DIST = 0.78

    CORNER = 0.25 * math.pi
    YAW_NOISE = 20.0 * math.pi / 180.0  # +/- 20 deg around corner-first

    def _candidates(n):
        # bearing of B relative to A, fully uniform: the frozen wrist means the
        # bearing of the slot matters, so it must be varied.
        phi = torch.rand(n) * (2.0 * math.pi)

        mode = torch.rand(n)
        n1 = (torch.rand(n) * 2.0 - 1.0) * YAW_NOISE
        n2 = (torch.rand(n) * 2.0 - 1.0) * YAW_NOISE
        free_a = torch.rand(n) * (0.5 * math.pi)
        free_b = torch.rand(n) * (0.5 * math.pi)

        # 65%: both cubes corner-first toward each other (tightest faces)
        # 17.5%: A corner-first, B free yaw
        # 17.5%: B corner-first, A free yaw
        d_a = torch.where(mode < 0.825, CORNER + n1, free_a)
        d_b = torch.where((mode < 0.65) | (mode >= 0.825), CORNER + n2, free_b)

        sup_a = s * (torch.abs(torch.cos(d_a)) + torch.abs(torch.sin(d_a)))
        sup_b = s * (torch.abs(torch.cos(d_b)) + torch.abs(torch.sin(d_b)))

        # target face gap: mostly very tight, a minority merely tight
        gap_hi = torch.where(
            torch.rand(n) < 0.85,
            torch.full((n,), 0.010),
            torch.full((n,), 0.025),
        )
        gap_lo = 0.0015
        gap = gap_lo + torch.rand(n) * (gap_hi - gap_lo)

        d = torch.clamp(sup_a + sup_b + gap, min=MIN_SEP + SEP_EPS)

        cx = X_LO + torch.rand(n) * (X_HI - X_LO)
        cy = Y_LO + torch.rand(n) * (Y_HI - Y_LO)

        ux = torch.cos(phi)
        uy = torch.sin(phi)

        a_x = cx - 0.5 * d * ux
        a_y = cy - 0.5 * d * uy
        b_x = cx + 0.5 * d * ux
        b_y = cy + 0.5 * d * uy

        yaw_a = phi - d_a
        yaw_b = phi - d_b

        a_xy = torch.stack([a_x, a_y], dim=-1)
        b_xy = torch.stack([b_x, b_y], dim=-1)
        return a_xy, b_xy, yaw_a, yaw_b

    def _valid(a_xy, b_xy):
        sep = torch.linalg.norm(a_xy - b_xy, dim=1)
        ok = sep >= (MIN_SEP + 0.5 * SEP_EPS)
        for xy in (a_xy, b_xy):
            ok = ok & (xy[:, 0] >= -0.2) & (xy[:, 0] <= 0.2)
            ok = ok & (xy[:, 1] >= -0.3) & (xy[:, 1] <= 0.3)
            dx = xy[:, 0] - BASE_X
            dy = xy[:, 1] - BASE_Y
            ok = ok & (torch.sqrt(dx * dx + dy * dy) <= MAX_BASE_DIST)
        return ok

    # ---- deterministic, guaranteed-valid fallback (varied by row index) ----
    idx = torch.arange(b, dtype=torch.float32)
    phi_f = (idx / float(max(b, 1))) * (2.0 * math.pi)
    d_f = 0.0600
    fb_a = torch.stack([-0.5 * d_f * torch.cos(phi_f), -0.5 * d_f * torch.sin(phi_f)], dim=-1)
    fb_b = torch.stack([0.5 * d_f * torch.cos(phi_f), 0.5 * d_f * torch.sin(phi_f)], dim=-1)
    fb_yaw_a = phi_f - CORNER
    fb_yaw_b = phi_f - CORNER

    a_xy = fb_a.clone()
    b_xy = fb_b.clone()
    yaw_a = fb_yaw_a.clone()
    yaw_b = fb_yaw_b.clone()
    filled = torch.zeros(b, dtype=torch.bool)

    for _ in range(12):
        c_a, c_b, c_ya, c_yb = _candidates(b)
        ok = _valid(c_a, c_b)
        take = ok & (~filled)
        take2 = take.unsqueeze(-1)
        a_xy = torch.where(take2, c_a, a_xy)
        b_xy = torch.where(take2, c_b, b_xy)
        yaw_a = torch.where(take, c_ya, yaw_a)
        yaw_b = torch.where(take, c_yb, yaw_b)
        filled = filled | ok

    # final hard safety: any row that somehow violates the floor gets the fallback
    good = _valid(a_xy, b_xy)
    good2 = good.unsqueeze(-1)
    a_xy = torch.where(good2, a_xy, fb_a)
    b_xy = torch.where(good2, b_xy, fb_b)
    yaw_a = torch.where(good, yaw_a, fb_yaw_a)
    yaw_b = torch.where(good, yaw_b, fb_yaw_b)

    z = torch.full((b, 1), 0.02)
    cubeA_xyz = torch.cat([a_xy, z], dim=-1).to(torch.float32)
    cubeB_xyz = torch.cat([b_xy, z], dim=-1).to(torch.float32)

    zero = torch.zeros(b)
    ha = 0.5 * yaw_a
    hb = 0.5 * yaw_b
    cubeA_quat = torch.stack([torch.cos(ha), zero, zero, torch.sin(ha)], dim=-1).to(torch.float32)
    cubeB_quat = torch.stack([torch.cos(hb), zero, zero, torch.sin(hb)], dim=-1).to(torch.float32)

    return dict(
        cubeA_xyz=cubeA_xyz,
        cubeA_quat=cubeA_quat,
        cubeB_xyz=cubeB_xyz,
        cubeB_quat=cubeB_quat,
    )
