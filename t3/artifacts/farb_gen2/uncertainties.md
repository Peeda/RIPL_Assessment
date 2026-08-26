## What I am least sure of

**1. The knife-edge gating on `is_cubeA_on_cubeB`.** That flag needs |Δz − 0.04| ≤
5 mm. A cube that is rocking will flicker it on and off, so my reward will
oscillate between ~4.4 (stage B) or ~1.4 (stage A) and ≥ 6.4 (stage D) on
consecutive steps. I chose the band edges so the discontinuity at the flag
boundary is small and always in the *upward* direction (stage B at the tolerance
boundary ≈ 4.4, stage D floor 6.4 — a 2.0 jump on release, which is intended, but
the A→D jump for a cube that is dropped from 1 cm up and lands on target is ~5.0
and is noisy). If PPO's value function struggles with that, the fix is to replace
the flag gate in stages C/D with a continuous soft gate on `xy_err` and `z_err`.
I did not do that because an ungated continuous "cube near stack pose" term is
exactly the thing that pays while the cube is still held.

**2. Whether `open_frac` is a real gradient.** For the Panda the finger-joint sum
is 0.04 while gripping a 40 mm cube and 0.08 fully open, so `open_frac ≈ 0.5`
*while grasping*. The baseline has the same property. It means stage C already
pays ~0.2 of its 0.4 release budget for doing nothing, and the residual only sees
0.2 of gradient for opening. If the residual's gripper channel is weakly
controlled by the frozen base policy this term may simply be inert; the real
release incentive is then the 2.0 band jump into stage D, which I believe is
enough.

**3. `upright` is the term I am least confident is well-scaled.** Cubes spawn flat
and mostly stay flat, so `upright` will read 1.0 in the vast majority of states
and contribute nothing; it only fires in the few steps where the cube is genuinely
perched. That is by design, but it means 0.30 of the placement budget is dead
weight most of the time, and the effective placement gradient is mostly
`xy_precision`.

## Which term an optimiser exploits first

**`clearance`** (weight 0.1, stage D). It is the only term that pays for moving the
tool *away* from the task, and the cheapest way to raise it is to yank the gripper
straight up immediately after opening — which at full arm extension is precisely
the motion most likely to catch a finger on the cube. I kept it at 0.1 (7 % of the
stage-D width, 1.25 % of REWARD_MAX) for that reason. If the validator sees
retreat-induced topples increase, delete this term; the rest of the reward stands
without it.

Second candidate: **stage B's `0.3·near_goal·low_motion`**. A policy that parks the
held cube 1–2 cm above cubeB and freezes gets ~4.5/step for 200 steps = 900
return, versus a stack completed at step 60 giving roughly 60·4 + 140·8 ≈ 1360. The
margin is real but not enormous; it is comparable to the built-in reward's own
margin (hold-at-goal ≈ 5.0/step vs success 8/step). If hover-and-hold emerges, the
fix is to cut stage B's width from 1.8 to 1.0.

## What I predict validation will find

* **Hand-built states.** Completed stack 8.0. Held-in-place-and-never-released
  ~6.0 (place_quality ≈ 1, open_frac ≈ 0.5) — a 2.0 margin, wider than the
  built-in reward's 1.25. Cube beside target on table ≤ 1.5. Target knocked away
  ≤ 1.5. Stacked wrong way round (Δz = −0.04, so `z_flag` false, not grasped)
  ≤ 1.5. Cube below the table at the right horizontal position ≈ 0 (tcp is far
  from it, so even stage A pays almost nothing). Order should be clean.
* **Monotone sweep.** Should pass. Moving cubeA away from the goal cannot turn on
  `placed`; within stage B the only distance-dependent terms are decreasing in
  `d_goal`; within stage A the only term is decreasing in `d(tcp, cubeA)`. The one
  thing I cannot verify without running it is what happens if the sweep keeps the
  gripper *at* the cube (so `d_tcp` stays constant while the cube leaves cubeB):
  then stage A is flat, not decreasing, at ~1.5. Flat is not increasing, so it
  should still pass, but if the harness requires strict decrease along the whole
  line this is where it will complain.
* **Rollout ranking at the failing stage.** This is the claim I would most like
  checked. My prediction: among episodes that all reach "released over cubeB",
  my reward separates the ones whose stack survives from the ones that topple
  more sharply than the built-in reward does, because the built-in reward has no
  positional term in that regime at all and mine has up to 1.0 of `place_quality`
  plus 0.3 of `low_motion`. Overall episode-return ranking will look similar to
  the baseline's (both are dominated by whether success was ever reached); the
  discrimination gain should show up specifically in the conditional comparison.
* **The reward cannot see history** (hacking note 7). "The stack was still
  standing at step 200" is not expressible; I approximate it with per-step
  "on target, released, flat, still" which is the same quantity only in
  expectation. A cube that is stationary for one step mid-topple briefly scores
  8.0. I do not have a way around this inside a stateless reward and am flagging
  it rather than faking it.
* **Sampler.** I expect ~1.2 % nominal capture to be confirmed, mean `d_B` ≈ 0.81,
  max ≈ 0.866, and zero constraint violations. The risk I see is that `d_A ≤ 0.70`
  is a slightly generous reading of "comfortable" — it admits cubeA at up to the
  ~75th percentile of the nominal distance distribution. I preferred that to a
  tight near band because collapsing cubeA into a small blob is the overfitting
  failure the brief warns about.
