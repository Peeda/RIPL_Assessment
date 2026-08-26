"""A hand-written sampler for mode `gap`, satisfying the contract. The positive
control for layer E, and the worked example the prompt points at.

The idea it demonstrates is the one T-II paid for: `face_gap` - the clearance
between the cubes' FACES along the bearing joining their centres - is the axis,
and centre separation is not. So rather than sampling two positions and hoping,
this samples the geometry directly:

    pick both yaws and the bearing  ->  the two cubes' half-extents along that
    bearing are then determined     ->  pick the FACE GAP you want
                                    ->  the centre separation follows.

That inversion is why the hit rate is ~1.0 instead of the nominal 4.6%.

The one subtlety, and it is the interesting one. The environment rejects any
placement closer than 58.6 mm centre-to-centre (stack_cube.py:88-90 feeding
samplers.py:63), so the smallest reachable face gap is
`0.0586 - extent_A - extent_B`. For two cubes presented square-on the extents
sum to 40 mm and that floor is 18.6 mm - barely inside mode `gap`'s 20 mm
threshold. For two cubes presented on their diagonals it is 2 mm. **Mode `gap`
is therefore mostly a statement about YAW**, not about proximity, and a sampler
that ignores yaw cannot reach most of the region however close it puts the
cubes. Computing the floor per row and sampling above it is what keeps every
draw physically legal without distorting the yaw distribution.
"""
import math

import torch

# The Panda's base (TableSceneBuilder), the table region and the placement
# floor. Constants rather than reads: the contract does not expose them on env,
# and inlining them here is what lets this file be checked without a simulator.
BASE_X = -0.615
BASE_Y = 0.0
CUBE_HALF = 0.02
MIN_SEP = 2 * (math.sqrt(0.02 ** 2 + 0.02 ** 2) + 0.001)
REACH_MAX = 0.80
DIST_MAX_CAP = 0.755          # mode `gap` requires dist_max < 0.76
GAP_LO, GAP_HI = -0.004, 0.018
TRIES = 12


def _half_extent(delta):
    """Half-width of a 40 mm square measured `delta` radians off its own axes.

    h*(|cos| + |sin|): 20 mm square-on, 28.3 mm across the diagonal.
    """
    return CUBE_HALF * (torch.abs(torch.cos(delta)) + torch.abs(torch.sin(delta)))


def sample_cube_poses(b, device):
    best_a = torch.zeros((b, 2))
    best_b = torch.zeros((b, 2))
    best_ta = torch.zeros(b)
    best_tb = torch.zeros(b)
    filled = torch.zeros(b, dtype=torch.bool)

    # Bounded `for`, never a `while`: a region that turns out to be hard to hit
    # must cost a fixed number of draws, not an unbounded loop inside the
    # environment's reset path.
    for _ in range(TRIES):
        theta_a = torch.rand(b) * (2 * math.pi)
        theta_b = torch.rand(b) * (2 * math.pi)
        psi = torch.rand(b) * (2 * math.pi)

        ext = _half_extent(psi - theta_a) + _half_extent(psi + math.pi - theta_b)
        # The smallest face gap this yaw/bearing combination can physically
        # reach, given the environment's own centre-separation floor.
        gap_floor = MIN_SEP - ext
        lo = torch.clamp(gap_floor, min=GAP_LO)
        gap = lo + torch.rand(b) * torch.clamp(GAP_HI - lo, min=0.0)
        sep = torch.clamp(gap + ext, min=MIN_SEP)

        # Place the pair around a midpoint drawn from the nominal support, then
        # keep only the rows that land legally.
        mid_x = torch.rand(b) * 0.30 - 0.15
        mid_y = torch.rand(b) * 0.50 - 0.25
        dx = torch.cos(psi) * sep * 0.5
        dy = torch.sin(psi) * sep * 0.5
        ax, ay = mid_x - dx, mid_y - dy
        bx, by = mid_x + dx, mid_y + dy

        d_a = torch.sqrt((ax - BASE_X) ** 2 + (ay - BASE_Y) ** 2)
        d_b = torch.sqrt((bx - BASE_X) ** 2 + (by - BASE_Y) ** 2)
        ok = ((ax.abs() <= 0.2) & (bx.abs() <= 0.2)
              & (ay.abs() <= 0.3) & (by.abs() <= 0.3)
              & (torch.maximum(d_a, d_b) < DIST_MAX_CAP)
              & (torch.minimum(d_a, d_b) > 0.20)
              & (d_a < REACH_MAX) & (d_b < REACH_MAX))

        take = ok & (~filled)
        best_a[take] = torch.stack([ax, ay], dim=1)[take]
        best_b[take] = torch.stack([bx, by], dim=1)[take]
        best_ta[take] = theta_a[take]
        best_tb[take] = theta_b[take]
        filled = filled | take

    # Any row still unfilled after the budget falls back to a legal default
    # rather than an invalid pose. Never return a row that violates the physics.
    if (~filled).any():
        n = int((~filled).sum())
        fallback_a = torch.tensor([0.0, -0.03]).repeat(n, 1)
        fallback_b = torch.tensor([0.0, 0.03]).repeat(n, 1)
        best_a[~filled] = fallback_a
        best_b[~filled] = fallback_b
        best_ta[~filled] = math.pi / 4
        best_tb[~filled] = math.pi / 4

    def _pose(xy, theta):
        xyz = torch.zeros((b, 3))
        xyz[:, 0:2] = xy
        xyz[:, 2] = 0.02
        quat = torch.zeros((b, 4))
        quat[:, 0] = torch.cos(theta / 2)      # w
        quat[:, 3] = torch.sin(theta / 2)      # z - rotation about z only
        return xyz, quat

    a_xyz, a_quat = _pose(best_a, best_ta)
    b_xyz, b_quat = _pose(best_b, best_tb)
    return dict(cubeA_xyz=a_xyz, cubeA_quat=a_quat,
                cubeB_xyz=b_xyz, cubeB_quat=b_quat)
