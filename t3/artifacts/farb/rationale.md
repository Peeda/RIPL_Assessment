## What the video actually shows

Grasp and transport are clean. The break is entirely at the end: the release
happens off-centre, or from ~5 mm too high, or while the cube still has sideways
velocity, at full arm extension where the wrist is least steady. The cube lands
on an edge, rocks, and slides off; sometimes the retreat clips it. In the last
frames the red cube is on the table *next to* the green one, not on it.

So the quantity to shape is **release and settle quality**, not reaching, not
grasping, not transport, and explicitly not "cubeB must not move" (the note says
clearance is not the issue and rewarding the absence of a symptom is the trap in
guidance #6).

## Stage ladder (bands are hard, non-overlapping)

| band | condition | shaping inside the band |
|---|---|---|
| **A: [0.0, 2.0]** | `~grasped & ~on` — including *the failure state*: cube dropped beside the stack | `2·g_goal·(0.65 + 0.35·f_tcp)` |
| **B: [2.5, 3.5]** | `grasped & ~on` (carry) | `0.6·align + 0.4·align·height` |
| **C: [4.0, 4.9]** | `grasped & on` (held at stack pose) | `0.45·open + 0.25·settle + 0.15·centre + 0.15·seated` |
| **D: [5.0, 6.5]** | `on & ~grasped & ~success` (released, not yet settled) | `0.35·centre + 0.35·settle + 0.30·upright` |
| **E: [7.0, 8.0]** | `success` | `0.45·centre + 0.30·upright + 0.25·clearance` |

Ordering: 2.0 < 2.5, 3.5 < 4.0, 4.9 < 5.0, 6.5 < 7.0. No shaping term can lift a
rung into the next band. Holding forever caps at **4.9**; letting go correctly
pays **≥ 7.0** — the letting-go incentive is a 2.1-point cliff, and no state the
policy can maintain without completing the task pays more than a completed stack
(guidance #1, #2, #3).

## Why each term

- **`g_goal = 1 − tanh(4·‖A − goal‖)`, all three axes**, goal = `(Bx, By, Bz+0.04)`
  — cubeA above cubeB, not the reverse (guidance #4, #5). A cube under the table
  at the right xy has large `d_goal`, so it scores near 0.
- **Stage A's reach term is *multiplied* by `g_goal`, not added.** This is
  deliberate: it makes stage A monotone non-increasing as the cube is moved away
  from the goal *regardless of where the hand is*. With an additive reach term,
  the "sweep the cube away from the target" test can increase the reward when the
  cube moves toward a retreated gripper. Checked analytically: `|d g_goal/dd|·0.65
  = 2.6 > 0.35·|d f_tcp/dd| ≤ 1.75`, so the product always decreases.
- **`carry = align + align·height`**: the height reward is gated on lateral
  alignment, so descending pays only once the cube is over the target. That shapes
  the *approach from above* that produces a square landing, and it never pays for
  coming down onto cubeB's shoulder.
- **Stage C** is the release stage and its dominant term is gripper openness, so
  the gradient points out of the band and into D. `settle` and `seated` there are
  the two things the video shows going wrong: releasing while still drifting
  sideways, and releasing from marginally too high.
- **`upright`** = max alignment of any body axis of cubeA with world z, mapped
  through `clamp((a−0.85)/0.15)`. Flat on a face → 1; balanced on an edge → 0.
  This is the one measurement that directly sees "landed on a corner and rocked",
  which the built-in reward has no term for at all.
- **`centre = 1 − tanh(40·d_xy)`**: sharp in the millimetre range, so the residual
  is paid for margin *inside* the 33 mm success tolerance rather than for merely
  clearing it. A stack centred to 2 mm scores 0.92, one at the tolerance edge
  0.19. Margin is exactly what survives being brushed.
- **`clearance`** appears only in stage E, where success is already true, so it
  cannot be farmed. It pays for having withdrawn the hand from the finished stack
  — the "arm's own retreat brushes it" mechanism.

**Where it beats the built-in reward.** The built-in pays `6 + (ungrasp+static)/2`
for anything satisfying `is_cubeA_on_cubeB`, and a flat 8 for success, so a
knife-edge stack at the tolerance boundary and a dead-centre stack are worth the
same, and a stack that will topple two steps later is worth 7.5 right now. Mine
grades success from 7.0 to 8.0 on centring, flatness and hand clearance, and
grades the pre-release hold on stillness and seating. That is the stage this
failure mode breaks at.

## Sampler

Failure region as described: **only the target is far**. I sample cubeB in polar
coordinates about the Panda base — radius `U(0.695, 0.775)`, bearing
`U(−0.36, 0.36)` rad — which sweeps the whole far arc of the reachable table
(roughly `x ∈ [0.03, 0.16]`, `y ∈ [−0.27, 0.27]`), well beyond the ~0.615 m
median of the nominal distribution but inside the 0.8 m IK ceiling. cubeA is
drawn radius `U(0.50, 0.685)`, bearing `U(−0.5, 0.5)` — comfortable, and always
radially inboard of the target, so every episode is a long outward transport.

Subset guarantee: the env's shared `xy` offset means its true support is *both
cubes in the box* **and** `|Δx| ≤ 0.2, |Δy| ≤ 0.4`. I enforce
`|x| ≤ 0.195, |y| ≤ 0.295, |Δx| ≤ 0.19, |Δy| ≤ 0.37`, so every draw is
reachable by the env's own sampler. Separation floor is 0.08 m, comfortably above
the 0.05857 m physical floor and matching "the two cubes are well clear of each
other" (this mode is not about neighbour interference). Yaw stays fully uniform
for both cubes — the frozen policy must still handle arbitrary yaw with a locked
wrist; biasing it would leak a different failure mode into this one.

Variety: two independent radii and two independent bearings plus two yaws — six
continuous degrees of freedom, no discrete modes, no corner collapse. Rejection
is a bounded `for` of 48 batched attempts with a hand-verified constant fallback;
per-attempt acceptance is ~60%, so the fallback is astronomically unlikely to be
returned.
