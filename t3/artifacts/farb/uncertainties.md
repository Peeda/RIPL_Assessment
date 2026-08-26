## What I am least sure of

**1. The cliff between stage A and stages D/E.** A released cube 3 mm outside the
xy tolerance scores ≤ 2.0; 3 mm inside scores ≥ 5.0. That discontinuity is what
makes the reward rank "placed and stayed" far above "placed and slid off" — the
exact thing being measured — but it means a residual that is *nearly* good enough
sees little local gradient in the failed state itself. The gradient that fixes it
lives in stages C and D (release stillness, seating, centring, uprightness),
which occur *before* the cube leaves the tolerance. I think that is the right
place for it, but if PPO learns slowly this is the first thing I would soften
(e.g. lift stage A's ceiling with a `g_goal`-gated near-miss bonus). I did not do
it pre-emptively because it costs ranking margin.

**2. `openness` in stage C is 0.5, not 0, while gripping.** The Panda's two
finger joints sit at ~0.02 each on a 40 mm cube, so `sum/width ≈ 0.5`. Stage C
therefore starts around 4.5 rather than 4.0 and the within-band gradient for
opening is only ~0.2 before the jump to D. The jump does the work; the term is
mostly there for continuity. If the validator's held-above-target state scores
~4.7 that is expected, not a bug.

**3. Term I expect an optimiser to exploit first: `clearance` in stage E.** It is
capped at 0.25 and gated on `success`, so it cannot be farmed without a standing
stack, but its literal reading is "after succeeding, get the hand away", and a
residual could learn a fast withdrawal that occasionally *causes* the knock it is
meant to prevent — success is re-evaluated every step, so a knock loses 5+ points
immediately, which should dominate. If validation shows late-episode success
flicker on this sampler, drop `clearance` and redistribute its weight to
`centre`. Second most exploitable: `upright`, which is 1.0 for a cube lying flat
*anywhere*, including flat on the table — harmless only because it is gated by
`is_cubeA_on_cubeB`.

**4. Statelessness (guidance #7).** "Was it still standing at the end" is an
episode-level property. I cannot express it. What I can express is the per-step
quality of the standing stack, and I am betting that integrating a high per-step
score over the tail is a good proxy. It is not the same quantity: a stack that
topples at step 199 collects almost the same return as one that never topples.
Also inexpressible: "released too fast" as a rate, and "the arm was at full
extension" as a difficulty measure — I deliberately did not put a base-distance
term in the reward, because paying more for far stacks rewards the sampler's
choice rather than the policy's behaviour.

**5. `seated` uses scale 200 (0.2 per mm).** That is aggressive and rests on my
assumption that sub-millimetre z error is meaningful given contact stiffness in
the GPU sim. If contact jitter is larger than ~2 mm, this term is mostly noise
and contributes 0.135 of band-C jitter. It cannot break the ordering.

## Predictions for the validation battery

- Hand-built states: completed stack 7.0–8.0 (dead-centre flat with hand clear
  ≈ 7.9); held-above-never-released ≈ 4.5–4.8; beside-the-target-on-the-table
  ≈ 1.3–1.9; target knocked away ≈ 0.2–1.0; stacked wrong way round ≈ 1.2–1.5;
  cube below the table at the right xy ≈ 0.0–0.3. Margin from stack to runner-up
  ≥ 2.2.
- Monotone sweeps: should pass in both directions. The vertical sweep from a
  completed stack goes 7.x → (leaves z tolerance) ~1.9 → decaying; the horizontal
  sweep 7.x → ~1.9 → decaying. The grasped-carry sweep decays smoothly via
  `carry`. The one I would watch is a sweep that moves the cube *toward a
  retreated gripper*: stage A is multiplicative precisely to survive that, and I
  checked the derivative, but if the sweep is done with the gripper closed on the
  cube (so `grasped` stays true across the whole line) it lives in band B, where
  `align` is monotone in `d_xy` and `align·height` in `dz` — also fine.
- 100 frozen-policy rollouts: I expect clean separation of success from failure,
  and — the part that matters — separation *within* the "reached, grasped,
  transported, released" subpopulation, because after release the two outcomes sit
  in different bands (≥5 vs ≤2) for the whole tail. The built-in reward also
  separates those, but less: it pays 8 vs a reach term, and it gives identical
  credit to a marginal and a well-centred stack, so I expect my advantage to show
  up as better ranking among the *near-misses*, not as a bigger overall gap.
- Failure mode I would flag if it appears: because roughly half of each episode is
  the static tail, per-episode mean reward is dominated by the terminal state.
  That is intended here, but it means the reward carries little information about
  *how* the placement was achieved, only about what it ended as.
