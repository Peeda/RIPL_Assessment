## What the video shows and what I am shaping

The two cubes start with almost no clear space between their nearest *faces*
(both turned corner-first, centres barely above the 58.6 mm rejection floor).
The wrist orientation is frozen, so the fingers meet the slot at an arbitrary
misalignment. The frames show the hand descending into the slot, the green cube
being shoved, and the episode then either failing to grasp or landing the red
cube nowhere useful. The engineer's note is explicit: the defect is in *approach
and descent/retreat*, not in settling.

So the behaviour I shape is **clearance of the gripper's own geometry from
cubeB while it is low enough to hit it**, plus **clearance of the carried cube
over cubeB's top face during transport**. I deliberately do *not* penalise cubeB
moving (guidance #6): the cheapest way to keep cubeB still is never to approach,
and that term would be farmed by hovering.

### Gripper-geometry clearance (the main new term)

An isotropic keep-out cylinder around cubeB would be either too loose to matter
or would push the TCP off cubeA at grasp time — in this region cubeA itself is
only ~59 mm from cubeB, so any radius large enough to describe a finger
collision also covers the grasp pose. Instead I model the three points that
actually collide: the TCP origin (palm/hand centre) and the two fingertips at
`tcp.p ± q_i · ŷ_tcp`, where `ŷ_tcp` is the tcp frame's y axis (the Panda's
finger prismatic axis) and `q_i` are the two finger joint values.

For each point:
* lateral penalty `clamp((0.045 − d_xy_to_cubeB)/0.025, 0, 1)` — zero at 45 mm
  from cubeB's centre (≈15 mm of true face margin, since cubeB's footprint
  radius is 20–28 mm and a finger is ~10 mm half-thick), saturating at 20 mm,
  which is inside the cube.
* height gate `clamp((z_topB + 0.012 − z_point)/0.030, 0, 1)` — zero above
  52 mm, i.e. clear over cubeB's 40 mm top face; full at or below cube-centre
  height. Being near cubeB while high is free; coming down into the slot is not.

`pen_grip` is the max over the three points. This is exactly the mechanism in
the video, and it leaves three escape routes available to a few-millimetre
residual: shift the TCP a little away from cubeB along the finger axis (grasp
cubeA on its far side), **narrow the aperture** (a 40 mm cube needs ~42 mm of
gap and the Panda opens to 80 mm, so up to 15 mm of finger swing per side is
pure waste — and because the penalty is evaluated at the *fingertips*, trimming
it is paid for automatically with no separate, gameable aperture term), or delay
the descent until laterally clear. All three are precisely the precision the
base policy lacks; none of them re-teach the task.

### Carried-cube clearance

`pen_cubeA = ov · vz`, with `ov = clamp((0.062 − d_xy(A,B))/0.022, 0, 1)` and
`vz = clamp((z_goal − z_A)/0.020, 0, 1)`, `z_goal = z_B + 0.04`. That is exactly
"cubeA is laterally close enough to touch cubeB *and* is still below cubeB's top
face". It is **identically zero at the goal pose** (vz = 0 there), so it cannot
fight the top of the ladder and cannot break the monotone sweep. It exists
because the built-in place term `1 − tanh(5‖goal − A‖)` is monotone in
straight-line distance and therefore rewards sliding the held cube in low and
sideways — ramming cubeB — which is the second half of this failure mode.

## The ladder (REWARD_MAX = 8.0, same scale as the built-in)

| stage | mask | band | shaping inside the band |
|---|---|---|---|
| 0 approach | `~grasped & ~on_B` | **[0.0, 2.0]** | `2·clamp(0.8·reach + 0.2·place − 0.5·pen_grip, 0, 1)`, reach = `1−tanh(5‖tcp−A‖)`, place = `1−tanh(5‖A−goal‖)`, both full 3-D |
| 1 carry / place | `grasped & ~on_B` | **[2.5, 4.5]** | `2.5 + 2·clamp(place − 0.5·max(pen_grip, pen_cubeA), 0, 1)` |
| 2 placed, still held or still moving | `on_B & ~success` | **[5.5, 6.5]** | `5.5 + 0.5·ungrasp + 0.5·static` |
| 3 completed | `success` | **8.0** | constant |

Strict ordering: 2.0 < 2.5, 4.5 < 5.5, 6.5 < 8.0. Every stage's shaping is a
`clamp(·,0,1)` times its span, so no rung can reach into the next. The grasp
jump is +2.5 (smaller than the built-in's +4 base) and the terminal bonus over
"placed but still held" is +1.5, so a grasp-and-hold policy is capped at
4.5/step against 8/step and releasing strictly dominates holding at every step.
The 0.2 weight on `place` in stage 0 gives a monotone pull toward the goal even
before the grasp and makes the "cube moved away from target" sweep better
behaved in an ungrasped state; it is capped at 0.4 out of 8, far below any rung
step. `ungrasp` and `static` are carried over from the built-in in spirit
(finger sum over gripper width, `1 − tanh(10v + ω)`) because the note says
settling already works and I have no reason to redesign the working part.

## Sampler

Configurations are built from the *face gap*, not the centre distance: bearing
uniform on the circle; each cube's yaw relative to that bearing (65% both
corner-first ±20°, 17.5% A corner-first with B free, 17.5% the mirror); support
computed per cube; target gap sampled (85% in [1.5, 10] mm, 15% up to 25 mm);
centre distance set to `max(sup_A + sup_B + gap, 0.05857 + ε)`. Because of that
clamp, even the face-on draws land near the environment's own floor. The pair
centre is uniform over x∈[−0.12, 0.09], y∈[−0.15, 0.15], which with a maximum
half-separation of ~0.04 guarantees both cubes stay inside x∈[−0.2, 0.2],
y∈[−0.3, 0.3] and within 0.78 m of the base. Bounded 12-iteration rejection with
a deterministic, per-row-varied fallback; no `while`, and every returned row is
re-validated before it goes out.

Concentration: essentially every draw has a corner-presented nearest face and a
gap under ~25 mm — a few percent of the nominal distribution. Variety: bearing
over the full circle, both yaws jittered, pair centre over a 21 cm × 30 cm
rectangle, a two-component gap mixture, and either cube may be the corner-first
one. Nothing collapses onto a single geometry or one corner of the table.
