## What I am least sure of

**1. The finger-axis assumption.** `pen_grip` places the two fingertips at
`tcp.p ± q_i · ŷ_tcp`, taking the tcp frame's **y** axis as the Panda's finger
opening direction and the tcp origin as roughly the fingertip plane. That is
what the URDF says (prismatic finger axis `[0,1,0]` in the hand frame,
`panda_hand_tcp` a pure z offset from `panda_hand`), but I cannot verify it from
the surface I am allowed to read. If the axis is wrong, the two extra points are
offset in the wrong direction and the lateral gradient becomes noise; the TCP
point (which is in the same max) still gives a correct, if blunter, keep-out, so
the term degrades rather than inverts. If validation shows the clearance term
doing nothing, this is the first thing to check — swapping to an isotropic
keep-out on `tcp.p` alone with radius ~0.055 is the fallback, at the cost of
fighting the grasp.

**2. A local valley in front of the grasp.** With the gripper fully open and the
finger axis pointing at cubeB, descending onto cubeA scores ≈0.9 in stage 0
while hovering 40 mm above it scores ≈1.55. Grasping is worth ≥2.5, so hovering
is not an optimum, and the residual is bounded so the base policy still drives
the descent — but this is the term I expect an optimiser to exploit first: a few
extra hover steps before committing, or a slightly shallow descent that
grazes-but-does-not-grasp. If validation shows stage-0 dwell time rising or the
grasp rate dropping relative to the built-in, lower the penalty weight from 0.5
to ~0.3 rather than removing the term.

**3. Whether the gripper channel of the residual is wide enough.** A large part
of the mechanism-correct fix is "descend with the fingers opened to ~50 mm
instead of 80 mm". That is expressible through the 4th action dimension, and my
reward pays for it implicitly, but if the residual's gripper component is
clipped to a very small band the reward is asking for something the policy
cannot deliver, and only the lateral-offset and stay-high escapes remain.

**4. Stateless limits.** I cannot see whether cubeB has *been* disturbed, only
where it is now. So "the stack landed off-target because the base was nudged
5 mm earlier" is invisible to me except through the goal moving with cubeB,
which lowers the place term as a side effect. That is an outcome signal, not the
behavioural one I want, and it is the reason the clearance terms exist at all.
I also cannot reward "approached from the far side of cubeA" as a path property;
only the instantaneous geometry.

## What I predict validation will find

* **Hand-built states.** Completed stack 8.0. Held above target and never
  released: 6.0–6.75 if it counts as `on_cubeB` (finger sum ≈ half of gripper
  width ⇒ ungrasp ≈ 0.5), or ≤4.5 if held clear of the 5 mm z window — either
  way ≥1.25 below the stack. Cube beside target on the table: ≤~1.9. Target
  knocked away with cubeA grasped: ~2.5–3.3. Wrong-way-round stack: ~0.9–1.9.
  Cube below the table at the right xy: ~0–0.3 (the place term is 3-D and the
  cubeA-low penalty saturates). Margin requirement should pass comfortably.
* **Monotone sweeps.** From the goal pose with the cube held, both the
  horizontal and vertical sweeps are strictly decreasing: `pen_cubeA` is exactly
  zero at goal height and only grows downward, and the fingertips sit above the
  height gate during placement. The one sweep I am not certain about is an
  *ungrasped* horizontal sweep with the TCP left stationary far from both cubes:
  there stage 0 is dominated by `0.8·reach`, and moving cubeA away from cubeB
  could move it toward the TCP and raise the reward. The built-in reward has the
  identical property (its non-grasped branch depends only on tcp–cubeA
  distance), so if this is flagged it should be flagged for both; the 0.2 place
  component makes mine slightly better behaved, not immune.
* **100 real episodes.** I expect the ranking gain over the built-in to come
  from stage 0/1 and not from the grasp indicator: episodes where the hand goes
  down through the slot with a finger inside cubeB's footprint accumulate a
  visibly lower stage-0/stage-1 return even before cubeB moves, and episodes
  that drag the held cube in low and sideways lose stage-1 shaping. The built-in
  gives all of those full credit. Where I expect *no* gain, and possibly a small
  loss, is on episodes that fail purely because the frozen wrist cannot line up
  at all — no amount of a few-millimetre clearance shaping rescues a
  configuration where the fingers cannot fit around cubeA without touching
  cubeB, and my reward will correctly but unhelpfully mark those episodes low
  throughout.
* **Sampler.** Target-region hit rate near 100% by construction; the risk is the
  opposite one — that some draws are *too* tight to be solvable by a bounded
  residual (gap ~2 mm with both corners facing), which would show up as a floor
  on the achievable success rate rather than as a training failure. If that
  floor is high, raise `gap_lo` from 0.0015 to ~0.004.
