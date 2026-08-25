#!/usr/bin/env python
"""What a failure mode IS, in terms of the initial state. Pure stdlib.

A T-II failure mode is a REGION OF THE INITIAL-STATE DISTRIBUTION, not a list of
bad episodes. That framing is what makes the claim testable: a region can be
resampled, so the rate that identified it and the rate that confirms it come
from disjoint episodes.

This file is the whole definition. It imports `math` and nothing else - no
numpy, no torch, no gymnasium - so it runs on a laptop with no ManiSkill
install, `t2/test_geometry.py` can check it in a second, and the analysis half
of the harness never needs a simulator. The sim half lives in `t2/harness.py`.

See t2/README.md for the method this sits inside.
"""
import math

# ---------------------------------------------------------------------------
# angles
# ---------------------------------------------------------------------------


def wrap(a):
    """(-pi, pi]. The one convention; see CLAUDE.md. Applied at log time, once."""
    return -((-a + math.pi) % (2 * math.pi) - math.pi)


def wrap90(a):
    """(-pi/4, pi/4].

    A cube has 4-fold yaw symmetry, so a relative yaw of +85 deg and one of
    -5 deg describe the SAME geometry. CLAUDE.md pins relative_yaw to (-pi, pi]
    and that column is logged exactly as specified - but it is the wrong axis to
    regress against, because it splits one physical configuration across two
    ends of the range. This is the axis analysis should use.
    """
    q = math.pi / 2
    return -((-a + q / 2) % q - q / 2)


def yaw(q):
    """Yaw from a wxyz quaternion, wrapped. Takes any 4+ element sequence."""
    w, x, y, z = [float(v) for v in list(q)[:4]]
    return wrap(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


# ---------------------------------------------------------------------------
# intervals
# ---------------------------------------------------------------------------


def wilson(k, n, z=1.96):
    """Wilson score interval. Returns (lo, hi); (nan, nan) for n == 0.

    Not sqrt(p(1-p)/n): at n ~ 100 the interesting bins sit near p = 0, where
    the normal interval runs below zero and claims certainty it does not have.
    A 0/20 bin gets +-0.000 from the normal formula and [0, 0.161] from Wilson.
    Same data, honest bars. Shared so every table in this harness quotes
    intervals computed the same way.
    """
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    lo, hi = c - h, c + h
    # At the extremes the bounds are exactly 0 and 1 analytically - for k = n,
    # c + h = (1 + z^2/n)/d = d/d = 1 - but computing c and h separately lands a
    # few ULP short, so min(1.0, .) leaves 0.9999999999999999. That is invisible
    # in a printed table and fatal in a plot: matplotlib rejects the resulting
    # -1.1e-16 error bar outright. Pin the exact cases.
    if k == n:
        hi = 1.0
    if k == 0:
        lo = 0.0
    return max(0.0, lo), min(1.0, hi)


# ---------------------------------------------------------------------------
# the geometry of an initial state
# ---------------------------------------------------------------------------

# The Panda's base, from TableSceneBuilder.initialize:
# mani_skill/utils/scene_builder/table/scene_builder.py:103 (panda) and :123
# (panda_wristcam, which is StackCube's default robot) both do
#   self.env.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
# Every reach distance below is measured from here. Getting this wrong once
# reversed the sign of a conclusion, so it is a named constant with a citation
# rather than an inline literal.
PANDA_BASE_XY = (-0.615, 0.0)

CUBE_HALF = 0.02                       # stack_cube.py:64,72 - 40 mm cubes

GEOM_FIELDS = ("face_gap", "dist_A", "dist_B", "dist_max", "dist_min")


def half_extent(delta, h=CUBE_HALF):
    """Half-width of an axis-aligned square of half-size h, measured along a
    direction `delta` radians away from its own axes.

    h*(|cos|+|sin|): h at 0 deg, h*sqrt(2) at 45 deg. This is the whole reason
    yaw matters - a 40 mm cube presents 56.6 mm across its diagonal.
    """
    return h * (abs(math.cos(delta)) + abs(math.sin(delta)))


def geom_features(ax, ay, tha, bx, by, thb, h=CUBE_HALF):
    """Clearance and reach, from the two cube poses alone. -> GEOM_FIELDS dict.

    `face_gap` is the free space between the two cubes' FACES along the line
    joining their centres, where `separation` is the distance between their
    CENTRES. The difference is the part that depends on yaw, and it is the part
    that predicts failure. Both yaws enter, each resolved along the A->B
    bearing - which is why `relative_yaw` on its own measures nothing: a yaw is
    not a scalar you can regress against without a direction to resolve it in.

        face_gap = separation - extent_A(bearing) - extent_B(bearing)

    It can go NEGATIVE, and that is meaningful rather than a bug: the two cubes'
    bounding squares overlap along the bearing between them, which the sampler's
    58.6 mm centre-separation floor does not exclude for diagonally-presented
    cubes. Do not clamp it.

    `dist_*` are measured from PANDA_BASE_XY, not from the world origin.
    """
    psi = math.atan2(by - ay, bx - ax)          # bearing from A to B
    sep = math.dist((ax, ay), (bx, by))
    dA = math.dist((ax, ay), PANDA_BASE_XY)
    dB = math.dist((bx, by), PANDA_BASE_XY)
    return dict(
        face_gap=sep - half_extent(psi - tha, h) - half_extent(psi + math.pi - thb, h),
        dist_A=dA, dist_B=dB,
        dist_max=max(dA, dB), dist_min=min(dA, dB),
    )


CUBE_FIELDS = (
    "cubeA_x", "cubeA_y", "cubeA_z", "cubeA_theta",
    "cubeB_x", "cubeB_y", "cubeB_z", "cubeB_theta",
    "separation", "relative_yaw", "relative_yaw_mod90",
) + GEOM_FIELDS


def cube_features(a_pose, b_pose):
    """Both raw_poses ([x,y,z,qw,qx,qy,qz]) -> CUBE_FIELDS dict."""
    a = [float(v) for v in list(a_pose)]
    b = [float(v) for v in list(b_pose)]
    ta, tb = yaw(a[3:7]), yaw(b[3:7])
    rel = wrap(tb - ta)
    out = dict(
        cubeA_x=a[0], cubeA_y=a[1], cubeA_z=a[2], cubeA_theta=ta,
        cubeB_x=b[0], cubeB_y=b[1], cubeB_z=b[2], cubeB_theta=tb,
        separation=math.dist((a[0], a[1]), (b[0], b[1])),
        relative_yaw=rel,
        relative_yaw_mod90=wrap90(rel),
    )
    out.update(geom_features(a[0], a[1], ta, b[0], b[1], tb))
    return out


def geom_from_row(r):
    """GEOM_FIELDS for a CSV row, recomputing them if the row predates them.

    Every column geom_features needs is in even the oldest CSV here, so the
    committed evidence base gains the geometry axes without being re-mined.
    """
    if all(k in r and r[k] != "" for k in GEOM_FIELDS):
        return {k: float(r[k]) for k in GEOM_FIELDS}
    return geom_features(float(r["cubeA_x"]), float(r["cubeA_y"]),
                         float(r["cubeA_theta"]),
                         float(r["cubeB_x"]), float(r["cubeB_y"]),
                         float(r["cubeB_theta"]))


# ---------------------------------------------------------------------------
# the two failure modes
# ---------------------------------------------------------------------------
#
# Defined ONCE. eval_modes.py selects seeds with them and re-asserts them
# per-episode against poses read out of the running env; verify.py re-checks
# them offline from the logged CSVs; report.py labels with them. Four consumers,
# one definition, so they cannot drift.
#
# Each mode carries exactly ONE control, and that control excludes the OTHER
# mode's factor. Without it the two regions overlap and neither rate is an
# independent measurement of its own mechanism. Verified disjoint over the
# 1,200-episode nominal pass: 0 episodes satisfy both.
#
# Thresholds are PRE-REGISTERED - written down here and in
# notes/t2-failure-modes.md before the confirmation passes ran, together with
# the rate each one has to reproduce on fresh seeds. A threshold fixed in
# advance is a materially stronger claim than one chosen after seeing the
# scatter.

MODES = {
    # A - the cubes' FACES are close. The gripper's orientation is frozen at
    # reset (pd_ee_delta_pos pads the action with three zeros and
    # compute_target_pose keeps the current rotation, pd_ee_pose.py:86-99), so
    # the policy cannot square up to a misaligned cube and the descent onto
    # cubeA fouls cubeB. Fails at PLACEMENT: it grasps normally and holds
    # normally once placed, but often never gets cubeA onto cubeB at all.
    #
    #   dist_max < 0.76  excludes mode B's far-reach factor.
    "gap": lambda g: g["face_gap"] < 0.020 and g["dist_max"] < 0.76,

    # B - cubeB is at the edge of the workspace, cubeA is comfortable. The arm
    # gets there: grasp rate 1.000, place rate near baseline. What collapses is
    # the stack STAYING - success needs cubeA on cubeB, static, AND released.
    #
    #   face_gap >= 0.05  excludes mode A's factor.
    #   dist_A < 0.72     excludes the both-cubes-far corner, where grasping
    #                     breaks down kinematically (0.815 above 740 mm) and no
    #                     bounded residual recovers a target the IK cannot
    #                     reach. Excluded rather than left in to depress the
    #                     result, since T-IV is scored on this region.
    "farb": lambda g: (g["dist_B"] >= 0.76 and g["dist_A"] < 0.72
                       and g["face_gap"] >= 0.05),

    # The reference arm. No filter - the nominal distribution, run in the same
    # 3 x 100 shape so the comparison is structural rather than assembled.
    # Also the arm T-IV needs for "near-zero degradation on the nominal
    # distribution".
    "nominal": lambda g: True,
}

# What the 1,200-episode discovery pass measured for each, so a confirmation
# pass on fresh seeds reads as confirming or not rather than merely reporting.
# (rate, wilson_lo, wilson_hi, n).
DISCOVERY = {
    "gap":     (0.523, 0.379, 0.662, 44),
    "farb":    (0.561, 0.410, 0.701, 41),
    "nominal": (0.713, 0.687, 0.738, 1200),
}


# ---------------------------------------------------------------------------
# the evaluation CSV schema
# ---------------------------------------------------------------------------
#
# Lives here, with the geometry and the modes, rather than with the script that
# writes it - because the CSV is the CONTRACT between the half of this harness
# that needs a simulator and the half that must run on a laptop. eval_modes.py
# writes exactly these columns in this order; verify.py asserts the header it
# finds is exactly this; report.py reads them. A schema declared in the writer
# is a schema the readers can only discover by failing.
COLUMNS = [
    "run_id", "mode", "block", "policy_seed", "seed",
    "cubeA_x", "cubeA_y", "cubeA_theta", "cubeB_x", "cubeB_y", "cubeB_theta",
    "separation", "relative_yaw", "relative_yaw_mod90",
    "face_gap", "dist_A", "dist_B", "dist_max", "dist_min",
    "success_once", "success_at_end", "ep_len",
    "ever_grasped", "ever_placed", "ever_static",
    "final_cubeA_x", "final_cubeA_y", "final_cubeA_z", "cubeB_displacement",
]


# ---------------------------------------------------------------------------
# seed blocks
# ---------------------------------------------------------------------------
#
# reset(seed=s) fully determines the initial state, so a seed is a lossless
# 8-byte handle on one episode's initial conditions and these ranges partition
# the evidence base. Two separate disjointness requirements, and the second was
# missed once already:
#
# 1. Passes must be disjoint FROM EACH OTHER. Re-measuring a region on the
#    rollouts that identified it measures noise, not a failure mode.
#
# 2. Every pass must be disjoint FROM THE DEMONSTRATIONS. Motionplanning demos
#    are generated from consecutive episode seeds starting at 0
#    (examples/motionplanning/panda/run.py:44-101), ~990 replayed and 800
#    trained on. Measured on the same checkpoint: 0.910 on seeds 0-299 against
#    0.713 on held-out seeds - a memorisation result, not a success rate.
RESERVED = {
    "demos":     (0, 1000),        # training initial states. NEVER evaluate here.
    "discovery": (1000, 2200),     # the 1,200-episode nominal pass (done)
    "t1":        (6000, 6300),     # the T-I 3 x 100 deliverable (done)
}

# Everything eval_modes.py draws starts above all of it, so the new passes
# cannot collide with anything already measured. The old harness selected from
# seed >= 2200 and needed ~5,000 eligible seeds to fill a region, which reached
# straight through the T-I block and picked up 20 of its seeds.
EVAL_BASE = 10000


def reserved_hit(seed):
    """Name of the reserved block `seed` falls in, or None."""
    for name, (lo, hi) in RESERVED.items():
        if lo <= seed < hi:
            return name
    return None
