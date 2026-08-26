## What the failure actually is

The policy reaches, grasps and transports fine. It also releases about as often as
anywhere. What fails is the *last 20 cm and the last 10 steps*: at full arm
extension the release happens off-centre, from slightly too high, or with residual
lateral velocity; the cube lands on an edge, rocks, slides or is brushed by the
retreat. So the reward must contain resolution in exactly one place — **placement
squareness, settling, and clean release** — and it must contain that resolution
*inside* the region where the built-in reward is flat.

That flatness is the concrete gap in the baseline. Once `is_cubeA_on_cubeB` flips
true, the built-in reward is `6 + (ungrasp + static)/2`: it contains **no positional
term at all**. A cube balanced on the very corner of the tolerance box (xy error
33 mm, z error 5 mm, tilted) scores identically to a cube sitting dead centre and
flat. Those two states have very different probabilities of still being a stack at
step 200. My reward separates them by up to 1.0 of reward, continuously, which is
the signal a few-millimetre residual can actually act on.

## Stage ladder (bands are strict and checked by reading)

| stage | gate | band | shaping inside the band |
|---|---|---|---|
| A approach | `~grasped & ~placed` | **0.0 – 1.5** | `1.5*(1 - tanh(5*d(tcp,cubeA)))` |
| B carry | `grasped & ~placed` | **3.0 – 4.8** | `1.5*(1-tanh(5*d(cubeA,goal)))` + `0.3*near_goal*low_motion` |
| C placed, still held | `placed & grasped` | **5.0 – 6.2** | `0.8*place_quality + 0.4*open_frac` |
| D placed, released | `placed & ~grasped` | **6.4 – 7.8** | `1.0*place_quality + 0.3*low_motion + 0.1*clearance` |
| E success | `success` | **8.0** flat | — |

`goal = cubeB.p + [0,0,0.04]`, i.e. cubeA **above** cubeB (checked against point 4
of the hacking notes; `d_goal` uses all three axes, so a cube at the right
horizontal position but the wrong height, including under the table, is far away).

Ordering: 1.5 < 3.0, 4.8 < 5.0, 6.2 < 6.4, 7.8 < 8.0. No shaping term can lift a
rung into the next band — every inner term is a bounded convex combination of
quantities in [0,1] with weights summing to the band width.

Anti-hoarding: the best a policy can do while still holding the cube is **6.2**,
and that requires the cube to already be sitting in the stacked pose with the
fingers wide; holding it in the air over cubeB caps at ~4.5. Letting go of a good
placement immediately jumps to ≥ 6.4 and to 8.0/step once it is static. Success is
flat, so there is no residual gradient that pays for fidgeting afterwards.

## The individual terms and the mechanism each addresses

* **`place_quality` = 0.50·xy_precision + 0.20·z_precision + 0.30·upright.**
  Scales are deliberately tight relative to the success tolerances:
  `xy_precision = 1 - tanh(xy_err/0.012)` (tolerance is 33 mm, so this is still
  falling steeply across the whole admissible band) and
  `z_precision = 1 - tanh(z_err/0.004)` (tolerance 5 mm). `upright` uses the
  largest |component| of the world-z row of cubeA's rotation matrix, clamped from
  0.8 to 1.0 — this is 1 whenever *any* cube face is flat (correct for a cube's
  six-fold face symmetry) and 0 beyond ~37° of tilt. That is precisely "landed on
  an edge or a corner and rocked". This is the term the baseline lacks.
* **`open_frac`** (finger-joint sum / full width) in stage C: gives a continuous
  path out of "held in the right place" toward release. It is capped at 0.4 so
  that opening without a good placement cannot outscore a good placement.
* **`low_motion` = 1 - tanh(10·v + ω)**, calibrated to `evaluate()`'s own static
  thresholds (1e-2 m/s, 0.5 rad/s). Used twice: with weight 0.3 in stage B only
  *near* the goal (so the arrival is calm rather than slow everywhere — a slow
  approach far away earns nothing extra), and with weight 0.3 in stage D, where it
  is the difference between "resting" and "sliding out of tolerance".
* **`clearance` = clamp((tcp_z − cubeA_top)/0.05, 0, 1)**, weight **0.1**, stage D
  only. This is the retreat-brush mechanism, shaped as *behaviour during the
  withdrawal* (go up, not sideways through the fresh stack) rather than as
  "penalise the cube moving" — which per hacking note 6 would be maximised by
  never approaching. It is only payable in a state that already has the stack
  made and released, so it cannot be farmed from a distance, and its weight is
  small enough that it cannot compete with the terms above it.

Nothing is paid per step for a maintainable non-progress state beyond the
approach/carry bands, which are strictly below the placement bands. Stage A tops
out at 1.5 (the baseline pays 2 for the same hover).

## Where the region is: derivation from `_initialize_episode`

Quantity chosen: **`d_B` = distance from the Panda base at (−0.615, 0) to cubeB**,
because the description names the target's distance from the base as the axis, and
explicitly excludes the both-cubes-far case.

Nominal distribution, from the source: `xy ~ U[−0.1,0.1]²` shared, plus per-cube
`UniformPlacementSampler` draws on `[[-0.1,-0.2],[0.1,0.2]]` with a 2·(0.02828+0.001)
= 0.05857 m separation floor. So each cube's x is triangular on [−0.2, 0.2] (σ ≈
0.082) and y is a trapezoid on [−0.3, 0.3], flat on [−0.1, 0.1] (σ ≈ 0.129). Hence
`d = sqrt((0.615+x)² + y²)` ranges over [0.415, 0.8685] with mode ≈ 0.615–0.63.

Integrating that density (arithmetic, not a guess):
`P(d ≥ 0.78) ≈ 2.7 %`, `P(d ≥ 0.80) ≈ 1.0 %`. Large `d` requires x ≳ 0.11, which
the triangular x density already makes rare.

So my region is `d_B ∈ [0.78, 0.866]` — the tail *starts* at the 97th percentile
and runs **all the way to the environment's own corner** (the true maximum is
0.86846; I stop at 0.866 purely so the validator's `≤ 0.868` check cannot fail on
floating point). I sample uniformly by area over
`{x ∈ [0.105, 0.2], |y| ≤ 0.3, 0.78 ≤ d ≤ 0.866}`, which is area-weighted toward
the far end (mean `d_B` ≈ 0.81) and spans both the far-corner draws (|y| ≈ 0.3) and
the far-straight-ahead draws (y ≈ 0, x ≈ 0.2). That variety matters: the two look
different to the arm and a residual trained on only one will not transfer.

cubeA is held **comfortable**: `d_A ≤ 0.70` (just above the nominal median, so it
spans the whole comfortable range from ~0.52 up), with separation ≥ 0.09 m
("well clear"; the env floor is 0.0586). Because cubeB is at `d ≥ 0.78` and cubeA
at `d ≤ 0.70`, the radial gap alone guarantees ≥ 0.08 m of clearance, so the
neighbouring-cube-disturbance mode is structurally excluded, as the description
requires.

**Fraction of nominal draws captured:** P(d_B ≥ 0.78) ≈ 2.7 %, times
P(d_A ≤ 0.70 | cubeB far) ≈ 0.45 (the shared offset that pushes cubeB out also
pushes cubeA out, so this is well below 0.5), times a near-1 factor for the
separation and joint-support constraints ⇒ **≈ 1.2 % of nominal episodes**. A thin
corner, as required.

**Subset of the env's support.** The env applies one shared `xy` offset, so a pair
is reachable iff both cubes are in one translated placement box: `|Δx| ≤ 0.2` and
`|Δy| ≤ 0.4` (plus both inside the box and separation ≥ 0.05857). I enforce
`|Δx| ≤ 0.195`, `|Δy| ≤ 0.39`, x ∈ [−0.2, 0.2], |y| ≤ 0.3, z = 0.02 exactly, yaw
uniform on [0, 2π) about z only — so every row has positive density under the
environment's own sampler. Rejection is two bounded `for` loops (24 and 40 draws,
first-accept so the accepted sample is uniform on the region) over an explicitly
constructed valid default, so no invalid row can be returned.
