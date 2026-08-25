# T-II: the failure regions, and the axes that actually predict them

Third debugging narrative in this repo, after `notes/pusht-detour.md` (a task
abandoned) and `notes/rgb-localisation.md` (a policy fixed). This one is about
measurement: the first production T-II pass ran clean and produced a T-I number
that was wrong by +0.20 for a reason that had nothing to do with the policy.

All numbers below: `best_eval_success_once.pt` from the 800-demo spatial-encoder
RGB run, `physx_cpu`, `pd_ee_delta_pos`, `max_episode_steps=200`, ManiSkill
`62ff3a5`, gymnasium 0.29.1, **NVIDIA RTX 4090**. 1,900 episodes in 55 minutes at
~2,000 episodes/hour, 10 subprocess workers. Wilson 95% intervals throughout —
at n≈100 near p=0 the normal interval runs below zero and claims certainty it
does not have.

---

## The T-I number was measured on the training set

The harness ran exactly as designed. `run_t2.sh` used three disjoint seed
blocks — `t1` on [0, 300), `mine` on [1000, 2200), `region` above that — and
disjointness was enforced and verified. Then the two passes disagreed:

| pass | seeds | `success_once` |
|---|---|---|
| `t1` (3 × 100) | 0–299 | **0.910** [0.872, 0.937] |
| `nominal` | 1000–2199 | **0.713** [0.687, 0.738] |

Same checkpoint. Same policy seed for `t1_seed1` against the nominal pass. About
seven standard errors apart.

Three innocent explanations, all checked and all dead:

- **Not the initial-state distribution.** Block 0–299 is if anything *harder*:
  P(sep < 80 mm) = 9.3% against the nominal block's 7.7%, mean separation 173 mm
  against 177 mm.
- **Not drift over the run.** The nominal pass is flat across all ten
  chronological blocks of 120 episodes (0.617–0.808, no trend).
- **Not a sick worker.** All ten `env_idx` values land in 0.63–0.78.

And the discriminating observation: **the gap is uniform across every separation
bin** — the `t1` block beats nominal by 0.15–0.35 in all seven of them. A
difference that survives conditioning on the failure axis is not a difference in
the failure axis. It is the seed block itself.

The cause is in ManiSkill's demo generator
(`examples/motionplanning/panda/run.py:44-101`):

```python
def _main(args, proc_id: int = 0, start_seed: int = 0):
    seed = start_seed          # 0
    ...
    res = solve(env, seed=seed, debug=False, ...)
    seed += 1                  # incremented on success AND on failure
```

**Motionplanning demos are generated from consecutive episode seeds starting at
zero.** The replay produced ~990 trajectories and training consumed 800 of them,
so roughly seeds [0, 1000) are initial states the policy was *trained on*. The
T-I pass evaluated the policy on its own training set.

The nominal block is clean, and that is checkable rather than assumed: its very
first bin (seeds 1000–1099) already sits at 0.680, and the block is flat from
there (1000–1299 → 0.693, 1300–2199 → 0.720). There is no step anywhere inside
it, so the demonstration range ends below 1000 and the step is entirely between
0–299 and 1000+.

### What it costs, and what it is worth

**T-I's reported number is 0.713 [0.687, 0.738]** on 1,200 held-out episodes,
not 0.910. The re-run on seeds 6000–6299 supplies the 3 × 100 the assignment asks
for, with its error bar.

The contaminated figure is not waste, though — it is a **memorisation
measurement** that nothing else in this project would have produced:

> Train 0.910 vs held-out 0.713 — a 0.197 generalisation gap on a policy trained
> from 800 motion-planned demonstrations, with the gap uniform across the whole
> difficulty range rather than concentrated in the hard cases.

That is a report paragraph on its own, and it is the kind of thing the 100→800
demo lever in `notes/rgb-localisation.md` predicts but never measured directly.

### Why the guard was in the wrong place

The harness enforced that the passes be disjoint **from each other**. Nobody
wrote down that they must also be disjoint **from the demonstrations**, because
the demonstration seeds were never thought of as occupying the same namespace as
the evaluation seeds — the demos are a file on disk, and seeds felt like a
property of the evaluation harness.

`run_t2.sh` now states both requirements in its seed-block header and
hard-refuses `T1_BASE < DEMO_SEED_CEILING`. The general lesson is narrower and
more useful than "check your splits": **when a harness derives episodes from
seeds, and the training data was also derived from seeds, the two are drawn from
one shared namespace and overlap is the default, not the exception.**

---

## The pre-registered failure mode: cubes too close

Separation < 80 mm, a threshold taken from the `UniformPlacementSampler` analysis
and **written into `CLAUDE.md` before any rollout ran**. The sampler enforces a
58.6 mm minimum centre separation for 40 mm cubes, which excludes overlap but not
the interesting regime; sub-80 mm leaves under 40 mm of clear space between
faces, about what the Panda's gripper needs to descend without fouling.

| | `success_once` | n |
|---|---|---|
| nominal, sep < 80 mm (discovery) | 0.533 [0.431, 0.631] | 92 |
| **targeted pass, sep < 80 mm (measurement)** | **0.580** [0.531, 0.627] | 400 |
| nominal, all separations | 0.713 [0.687, 0.738] | 1200 |

Two independent measurements on disjoint seeds, agreeing inside their intervals.
The targeted pass is 200 fresh seeds drawn by rejection sampling over the seed
index, all ≥ 2212, each run twice.

**There is a dose-response inside the region**, which is what separates a real
region from a threshold that happened to land somewhere:

| separation | `success_once` | n |
|---|---|---|
| < 62 mm | 0.346 [0.232, 0.482] | 52 |
| 62–66 mm | 0.412 [0.311, 0.522] | 80 |
| 66–70 mm | 0.648 [0.544, 0.739] | 88 |
| 70–74 mm | 0.667 [0.547, 0.768] | 66 |
| 74–80 mm | 0.702 [0.612, 0.778] | 114 |

At the sampler's own floor the policy is worse than a coin flip; by 80 mm it has
recovered to near the unconditional rate.

### It is a property of the state, not of the sampler

Diffusion sampling is stochastic, so a seed's outcome is a draw rather than a
property. The two repeats settle how much of the region is real:

**148 of 200 seeds gave the same outcome twice (0.74)**, where 0.50 is what pure
policy noise would give and 1.00 is what a purely deterministic difficulty would
give. Most of the variation is the initial state. The two repeats also agree with
each other as independent estimates — 0.610 [0.541, 0.675] and 0.550 [0.481,
0.617].

### The mechanism, not just the correlation

`cubeB_displacement` (‖final − initial‖ over cubeB's xy) was logged specifically
to test the hypothesised mechanism — that the gripper's descent onto cubeA fouls
cubeB — rather than to describe the outcome:

| region episodes | cubeB moved (mean) | (median) | n |
|---|---|---|---|
| **failures** | **29.3 mm** | 19.5 mm | 168 |
| successes | 3.0 mm | 0.1 mm | 232 |

An order of magnitude, and it points the right way. Against a 40 mm cube, a 29 mm
displacement is the target being shoved most of its own width. The failure
taxonomy agrees: `grasped, never placed` rises from 10.0% of nominal episodes to
17.8% in the region — the policy gets cubeA off the table and then has nowhere
correct to put it, because cubeB is no longer where it was seen.

### It survives the obvious confounder

Separation correlates with how far the cubes sit from the robot in the nominal
distribution (mean max-reach rises from 150 mm to 234 mm across the separation
range), so the two axes have to be disentangled. Holding reach distance in its
middle two quartiles:

| separation | `success_once` | n |
|---|---|---|
| < 80 mm | **0.604** [0.463, 0.730] | 48 |
| 80–140 mm | 0.832 [0.773, 0.878] | 196 |
| 140–200 mm | 0.814 [0.746, 0.866] | 161 |

The deficit is undiminished. This is a separation effect, not a reach effect in
disguise.

---

## The second failure mode: cubes too far apart

**Exploratory, not pre-registered.** It was found by slicing the nominal pass
after the fact, which is a materially weaker claim than mode 1's, and the
targeted resampling pass that would turn it into a measurement is still pending.
The numbers below are from the discovery data and are labelled as such.

`CLAUDE.md` did hypothesise both tails in advance — *"too close and the grasp
approach on A collides with B; too far and the place phase runs out of steps"* —
but only the near threshold was given a number, so only the near mode counts as
pre-registered.

From the nominal pass, with reach distance held in its middle two quartiles:

| separation | `success_once` | n |
|---|---|---|
| 80–140 mm | 0.832 [0.773, 0.878] | 196 |
| 140–200 mm | 0.814 [0.746, 0.866] | 161 |
| 200–260 mm | 0.771 [0.682, 0.841] | 105 |
| **260–440 mm** | **0.689** [0.587, 0.775] | 90 |

So the success-versus-separation curve is a **U**, and the two failure modes are
the two tails of one pre-registered axis. Sub-260 mm prevalence in the nominal
distribution is 16.7%, so the far tail is about twice as common as the near one.

**The stated mechanism is already refuted, and the note should say so.** "The
place phase runs out of steps" predicts late grasps and truncated episodes. The
median first-grasp step is **46 in every separation bin** — 0–80 mm through
260–440 mm, no trend. Whatever the far mode is, the policy is not running out of
clock on the way to the cube. The mechanism is open, and the stride-1 trace from
the targeted pass is what should close it.

### A correction that changed the answer

The first version of this analysis measured reach distance as `hypot(x, y)` from
the origin and concluded that cubes *far* from the robot fail. That was wrong:
`panda_wristcam`'s base sits at `[-0.615, 0, 0]`
(`table/scene_builder.py:104`), not at the origin.

Recomputed from the correct base, **the direction reverses** — it is cubes *near*
the base that fail:

| max reach from base | `success_once` | n |
|---|---|---|
| 440–480 mm | 0.273 [0.097, 0.566] | 11 |
| 480–520 mm | 0.579 [0.363, 0.769] | 19 |
| 520–560 mm | 0.671 [0.555, 0.770] | 70 |
| 600–700 mm | 0.767 [0.731, 0.800] | 572 |

Physically sensible — the Panda's workspace is cramped close in. It was rejected
as the second mode on sample size (30 episodes below 520 mm, 11 in the worst bin)
rather than on effect size, and it is a live candidate if the far-separation
resample disappoints.

The lesson is cheap and general: **a derived geometric feature is a modelling
assumption, and it needs a source citation like any other.** `hypot(x, y)` looked
like arithmetic rather than a claim about where the robot is, so it went
unchecked, and it produced a confident conclusion pointing the wrong way.

---

## The failure taxonomy

Free from the per-step flags, and it is what turns a binary outcome into a
three-way diagnosis at zero extra cost.

| | nominal (held out) | region, sep < 80 mm |
|---|---|---|
| success | 71.3% | 58.0% |
| placed, then toppled | 16.1% | 19.2% |
| grasped, never placed | 10.0% | **17.8%** |
| never grasped | 1.6% | 3.8% |
| placed, never released/settled | 1.0% | 1.2% |

The region does not fail *differently* so much as it fails *more*, with one
exception: `grasped, never placed` nearly doubles, which is the signature of the
displaced-cubeB mechanism above.

**`placed, then toppled` is the largest failure class overall** at 16.1% of all
nominal episodes, and it is the same phenomenon as the `success_once` vs
`success_at_end` gap (0.713 vs 0.631, **+0.083**). It is a genuine failure mode
and `CLAUDE.md` flagged it as a candidate — but it is defined by *outcome*, not
by a region of the initial-state distribution, and T-II asks for regions. It is
recorded here as the strongest remaining candidate if a third mode is ever
wanted; finding which initial states produce it is unfinished work.

---

## Refinement: separation was the wrong axis, twice

Everything above is measured. Nothing in it is retracted. But watching the
rollouts raised two specific doubts, and testing them against the 1,200
held-out episodes already in hand — no new data — showed that **centre
separation is a proxy for two different things, and it is the worse
measurement of both.**

### Mode 1 is really about the gap between cube *faces*

The doubt: in the close-separation videos the gripper does not simply run out
of room. It comes down at an angle that is not square to the red cube's faces,
and a finger clips the green one.

That has a mechanism in the source, and it is stronger than the video
suggested. Under `pd_ee_delta_pos` the action is padded with three zeros and
`compute_target_pose` "keeps the current rotation"
(`pd_ee_pose.py:86-99`), so **the gripper's orientation is frozen at its reset
value for the whole episode.** The policy cannot rotate the wrist to square up
to a cube; it has one approach angle and the cube's yaw is uniform on [0, 2π)
(`random_quaternions(bounds=(0, 2π), lock_x, lock_y)`). A 40 mm cube gripped
near 45° presents 40·√2 = **56.6 mm**.

So the quantity that matters is not the distance between cube centres but the
clearance between their **faces** along the line joining them:

```
face_gap = separation − extentA(bearing) − extentB(bearing)
extent(δ) = 0.02·(|cos δ| + |sin δ|)
```

Both yaws enter, through the A→B bearing. Measured on the nominal pass:

| face gap | success | n |
|---|---|---|
| 5–20 mm | **0.500** | 48 |
| 20–40 mm | 0.604 | 111 |
| 40–70 mm | 0.749 | 175 |
| 70–120 mm | **0.794** | 310 |

And it is not just separation relabelled — **within `separation < 100 mm` the
low-gap half succeeds 0.524 against the high-gap half's 0.670.** Centre
separation was discarding signal it already had.

**This also rescues a null result reported above.** The taxonomy section records
relative yaw as having no effect, and that is correct as stated and misleading
as a conclusion. Relative yaw *alone* predicts nothing. Yaw *combined with the
bearing* — which is what `face_gap` does and `relative_yaw` cannot — predicts a
great deal. The right lesson is not "yaw does not matter" but "yaw is not a
scalar you can regress against on its own."

### Mode 2 is mostly a reach mode wearing a separation costume

The second doubt: in the far-separation videos, one cube looks like it might be
at the edge of what the arm can get to.

Distances here are from the **Panda's base at (−0.615, 0)**
(`table/scene_builder.py:103,123`), not the world origin. Against `dist_max` —
the distance to the *farther* of the two cubes — success is an **inverted U**:

| `dist_max` | success | n |
|---|---|---|
| 400–520 mm | **0.467** | 30 |
| 520–600 mm | 0.673 | 208 |
| 600–680 mm | **0.774** | 464 |
| 680–760 mm | 0.754 | 366 |
| ≥ 760 mm | **0.508** | 132 |

The two effects are not the same effect. The 2×2 on the nominal pass:

| | `dist_max` < 760 mm | ≥ 760 mm |
|---|---|---|
| separation < 260 mm | 0.752 (n=906) | 0.553 (n=94) |
| separation ≥ 260 mm | 0.667 (n=162) | 0.395 (n=38) |

Reach costs roughly twice what the separation tail does, and they stack. **The
"cubes too far apart" mode above is largely this one, seen through a correlated
variable** — a large separation makes it likelier that at least one cube is far
from the base. The original number is not wrong; the axis is.

### The two ends of the reach band fail *differently*

This is what makes them separate modes rather than one curve. Base rates on the
nominal pass are `never grasped` 0.016 and `grasped, never placed` 0.100:

| band | success | never grasped | grasped, never placed |
|---|---|---|---|
| `dist_min` < 520 mm | 0.615 | 0.019 | **0.178** |
| mid band | 0.770 | 0.005 | 0.069 |
| `dist_max` ≥ 760 mm, far cube = **A** | 0.478 | **0.149** | 0.179 |
| `dist_max` ≥ 760 mm, far cube = **B** | 0.538 | 0.015 | 0.169 |

Read across: when **cubeA** is the far one, the robot fails to grasp it at 9×
the base rate — it cannot get there. When **cubeB** is the far one, grasping is
completely normal and the failure is downstream, at the place step. And the
**near-base** band grasps normally too and also fails at placement, at 2.6× the
mid-band rate, at the *opposite* end of the same axis.

Three different mechanisms, one geometric feature, all separated by flags that
`evaluate()` already returns.

### Toppling is a consequence, not a mode

The taxonomy section flags `placed, then toppled` as the strongest remaining
candidate for a third mode. It is not one, and this closes that question.

It is real and large — 28.7% of the episodes that get cubeA onto cubeB do not
still hold at step 200 — but **no initial-state feature predicts it.** Binned by
`face_gap`, `separation`, `dist_min` or `dist_max`, the hold rate is flat. T-II
asks for regions of the initial-state distribution, and this is not one.

What does predict it is how far cubeB got shoved during the episode:

| cubeB displacement | hold rate | n |
|---|---|---|
| < 0.1 mm | **0.919** | 591 |
| 0.1–1 mm | 0.720 | 175 |
| 1–5 mm | 0.553 | 123 |
| 5–20 mm | 0.218 | 78 |
| > 20 mm | **0.032** | 94 |

That is an *outcome*, not a state — so it cannot define a region. But it says
what toppling is: **the downstream consequence of the face-gap mode.** Disturb
the base cube on the way in and the stack does not survive. Which also means a
fix for mode 1 should show up in `success_at_end`, not only in `success_once`.

### Is the far region even the policy's fault?

A region the task itself cannot solve is not a policy failure mode, and quoting
it as one would be wrong. Two checks, in order of cost.

**The trivial reading is already dead.** Over 8,000 indexed seeds `dist_max`
reaches 839 mm; the sampler's theoretical corner is 869 mm; the Panda's spec
reach is 855 mm. `P(dist_max > 855 mm) = 0`. **No cube is out of reach in the
free-orientation sense.**

But spec reach assumes the arm may pick any wrist orientation, and this one
cannot — see the frozen gripper above. `t2/reach_map.py` measures the reachable
set for that fixed orientation by asking the controller's own `Kinematics`,
confirming each solution by forward kinematics rather than trusting a local IK
solver's return value. `t2/demo_feasibility.py` gets an independent second
opinion from the motion planner, which has full state access and no perception
problem: demo seeds run consecutively from 0 and failed attempts are dropped,
so **every gap in the recorded seed sequence is a state the planner could not
solve.** Both are written; neither has run.

The likely answer is neither "impossible" nor "fine": reachable in principle,
but near full extension the Jacobian is ill-conditioned, so a `delta_pos`
command of a given size buys less accurate Cartesian motion. "The policy
degrades near the kinematic boundary" is a legitimate finding — and a more
interesting one than either extreme.

### The far-cubeB failure is not a reach failure

An earlier draft of this section argued that the far region could not be a T-IV
target, because `a = a_base + clip(Δ, −α, α)` bounds Δ in metres and a bounded
translation cannot extend the robot's reach. **That argument was wrong**, and
the data that killed it was already above — it just needed conditioning on the
right thing.

Split the far region by *which* cube is far, and condition on the other one
being comfortable so only one distance varies:

| cell | n | success | grasp | place given grasp | **hold given place** |
|---|---|---|---|---|---|
| both < 720 mm (reference) | 900 | 0.736 | 0.992 | 0.904 | 0.820 |
| only cubeA far (B < 720) | 48 | 0.542 | 0.917 | 0.773 | 0.765 |
| **only cubeB far (A < 720)** | 46 | 0.565 | **1.000** | 0.870 | **0.650** |
| both ≥ 720 mm | 58 | 0.483 | 0.879 | 0.824 | 0.667 |
| both ≥ 740 mm | 27 | 0.296 | **0.815** | 0.773 | 0.471 |

Read the far-cubeB row: the robot grasps **every time**, and it gets cubeA onto
cubeB 87% of the time against a 90% reference. It is not failing to reach
anything. What collapses is `hold | place` — 0.650 against 0.820. **The stack is
built and does not survive.** That is a precision-and-settling problem, which is
exactly what a small bounded translation correction is for.

The original argument was also wrong in kind: **α bounds the *per-step* delta**,
applied over ~200 steps, so a persistent correction accumulates. The binding
constraint is not α's magnitude but the IK saturating, and that only happens at
the arm's true kinematic edge — which in this data is the both-cubes-far corner,
where grasp drops to 0.815 (n=27). That corner is real and is genuinely beyond a
residual's reach. It is excluded from the mode rather than left in to depress
it, which is why the region is defined as *cubeB far **and cubeA comfortable***.

**A correction to the mechanism table above.** The `never grasped = 0.149` for
"far cube = A" is contaminated: `far_is_B` only says which cube is farther, so
that cell is dominated by episodes where *both* are far. Isolating cubeA ≥ 760 mm
with cubeB < 720 mm gives 0.083 — still 5× the base rate, but half of what the
earlier table implies, and the grasp failures actually concentrate in the
both-far corner rather than tracking cubeA's distance. The general lesson:
**when attributing a mechanism to one cube, condition on the other**, or a
"which cube" split silently measures "how many cubes".

### The two modes do not share a mechanism

Worth checking, because if they did there would only be one mode:

| cell | n | median cubeB displacement | >5 mm | hold given place |
|---|---|---|---|---|
| reference (both < 720, gap > 50 mm) | 715 | 0.04 mm | 0.166 | 0.830 |
| **mode A**: `face_gap` < 25 mm | 69 | **0.69 mm** | **0.377** | 0.809 |
| **mode B**: cubeB far, cubeA close | 41 | 0.37 mm | 0.195 | **0.657** |

Mode A disturbs the base cube on the way in and its stacks hold normally once
built. Mode B barely disturbs anything and its stacks fall over. Different
stages, different mechanisms, one shared axis-family — which is a much better
pair to report than two slices of the same curve.

### What this changes about the deliverable

The two headline modes are now **face gap** and **near-base**, and they are
chosen to be robust to however the feasibility question lands: both are
comfortably inside the reachable set, and the motion planner demonstrably solves
states in both. The far-reach region is reported as a third finding whose
interpretation waits on the checks above.

Pre-registered thresholds, fixed before any of the three passes ran, with the
nominal-pass prediction each has to reproduce on fresh seeds:

| pass | region | predicts | role |
|---|---|---|---|
| `gap` | `face_gap < 25 mm`, reach controlled | 0.640 [0.501, 0.759] | **T-IV target A** |
| `farb` | `dist_B ≥ 760 mm`, `dist_A < 720 mm`, gap controlled | 0.561 [0.410, 0.701] | **T-IV target B** |
| `nearbase` | `dist_min < 520 mm`, gap and far-reach controlled | 0.637 [0.564, 0.704] | third T-II finding |

against 0.713 unconditional. The three filters partition on `dist_min` and
`dist_max`, so they are mutually exclusive and each controls for the others'
factor — without that they contaminate each other and none of the numbers means
anything. `bash t2/run_modes.sh`.

### Why "3 seeds" means 3 seed *blocks*

The per-mode deliverable is 100 rollouts × 3 seeds. There are two ways to read
that, and they are not close to equivalent — one of them reports an error bar
about half the size it should be.

Write an episode outcome as `Y(θ, ω)`: θ the initial state, ω the DDPM sampling
noise. Let `q(θ) = E_ω[Y]` be how hard that particular state is. Outcome
variance splits in two, and the region passes' repeats measure the split
directly — agreement is **0.740** where pure coin-flip noise at p = 0.58 would
give 0.513:

| source | variance | share of p(1−p) = 0.2436 |
|---|---|---|
| between states, `Var(q)` | 0.1136 | **47%** |
| within a state (DDPM), `E[q(1−q)]` | 0.1300 | 53% |

Now compare, at 300 episodes either way:

- **A — three disjoint blocks of 100, one policy seed each.** Three independent
  estimates. SE of the mean `sqrt(p(1−p)/300)` = **0.0285**, and the observed
  spread reflects it.
- **B — one block of 100 under three policy seeds.** Every run contains the same
  term `(1/100)·Σ q(θᵢ)`, the average difficulty of *those* 100 states. It is
  common to all three and so contributes **nothing** to their spread. True SE is
  `sqrt(Var(q)/100 + E[q(1−q)]/300)` = **0.0396** — worse, because only 100
  distinct states are ever seen — while the observed spread gives
  `SD/sqrt(3)` = **0.0208**, understating its own SE by **1.9×**.

So B is less precise *and* claims to be more precise. The floor is the giveaway:
`Var(q)/100 = 0.0337` is irreducible by re-running the policy, yet B would
report 0.0208. **Re-running the policy on the same states cannot reduce
uncertainty about which states you drew, but the spread will look as though it
did.**

The 47/53 split is what makes this bite. Were DDPM noise 95% of the variance, B
would be nearly harmless.

None of which says policy-seed variation is uninformative — it measures
`E[q(1−q)]`, which answers "is this region hard, or is the policy erratic here?"
That is a real T-II claim and it is why the near/far passes ran repeats
(148/200 seeds repeated their outcome). It is simply a different quantity from
the error bar, and the deliverable asks for the error bar.

Using disjoint blocks costs nothing for T-IV: all 300 seeds are the fixed
targeted-evaluation set, re-run in the same blocks under the same policy seeds,
so the before/after is still paired at the state level — with three times the
state coverage. `t2/verify.py` asserts the shape.

### What I would do differently

- **Derive the geometry before mining, not after.** `face_gap` and the reach
  distances need nothing that was not already logged; both were recoverable from
  the committed CSVs in an afternoon. Had they existed at log time, the first
  targeted pass would have gone after the right region.
- **Read the controller before choosing a feature.** The frozen wrist is four
  lines of `pd_ee_pose.py` and it is the entire reason yaw matters. It was
  cheaper to read than the run that motivated reading it.
- **A null result is a result about a feature, not about a variable.** "Relative
  yaw does nothing" was recorded as though yaw were irrelevant. It was a
  statement about one particular scalar.

---

## Still open

- **The three confirmation passes.** `bash t2/run_modes.sh` — 1,200 episodes,
  ~37 min at the measured 2,000 ep/h, stride-1 traces. Until they land, the
  refined numbers above are post-hoc slices of the nominal pass and must be
  labelled that way, exactly as the far-separation mode was before its own pass.
- **The two feasibility checks.** `t2/reach_map.py` (needs the pod; policy-free)
  and `t2/demo_feasibility.py` (a metadata join, minutes). These decide whether
  the far-reach region is reported as a policy failure or as a task artifact.
- **The backend agreement check.** `bash t2/run_t2.sh backend` — replays the
  nominal pass's exact initial states on `physx_cuda` and compares per episode.
  It cannot pair by seed: cube poses come from `torch.rand` on the sim device
  and `reset()` seeds the whole batch from `_episode_seed[0]`, so the same seed
  is a different state on GPU and, at width > 1, not even one episode. This is
  what licenses T-IV training on GPU and evaluating on CPU. Judge the agreement
  against the same-backend floor the repeats measure (0.74 near, 0.67 far), not
  against 1.0 — DDPM sampling is stochastic and identical states disagree
  CPU-vs-CPU too.
- **Videos.** `analyze` prints a shortlist by rule; the recording pass retries
  until the outcome class matches. Not yet run. Worth recording from the region
  seed lists rather than the nominal shortlist, so each clip illustrates a
  named mode.
- **The state policy on the same seed lists.** "Do both policies fail in the same
  region?" separates a perceptual failure from a geometric one for the cost of a
  re-run. The refined axes make this sharper than it was: a *geometric* mode
  should reproduce on the state policy, and a *perceptual* one should not.

---

## What T-IV should target

**Two targets**, `gap` and `farb`, chosen because they fail at different stages
by different mechanisms — see the two tables above. Two modes that break in
genuinely different ways is a stronger T-II result than two slices of one curve,
and it gives T-IV two independent chances to show an effect.

Both are residual-fixable:

- **`gap`** is an approach-precision problem. A few millimetres of lateral
  correction during the descent is exactly what `clip(Δ, −α, α)` supplies, and
  it has the best-attested mechanism (cubeB displaced 29.3 mm in failures vs
  3.0 mm in successes). Because toppling tracks that displacement, a fix should
  show up in `success_at_end` as well as `success_once`.
- **`farb`** is a settling problem, not a reach problem — the robot grasps every
  time and places 87% of the time, and what fails is `hold | place` at 0.650
  against 0.820. Also precision, at a different stage.

The one place a bounded residual genuinely cannot help is the both-cubes-far
corner, where the IK saturates and grasp falls to 0.815. That is excluded from
`farb` by construction rather than left in to depress the result.

**If the budget will only carry one**, drop `farb` and say so. `gap` has the
better-attested mechanism and the larger sample. The standing warning applies:
one mode with clean curves and a proper 3-seed eval on both distributions beats
two half-trained runs.

**Never cut** the nominal-distribution re-evaluation, for either target. A
residual that fixes its region and degrades everywhere else is not an
improvement, and at 3 × 100 rollouts "near-zero degradation" is a claim the data
supports only weakly — say so.


---

## The harness was rebuilt around this result (2026-08-25)

Everything above is the *discovery* arc: the axes were found by slicing a
1,200-episode held-out pass, and the harness that produced it grew one script at
a time while the question was still open. Once the answer was two modes, that
shape stopped earning its keep — three shell drivers calling each other, a
generic `--where` seed selector, a 552-line analysis monolith, and four
diagnostic scripts whose questions were settled.

It was collapsed to one driver (`t2/run.sh`) and one deliverable script
(`t2/eval_modes.py`). The method is written up in **`t2/README.md`**, which is
now the place to start.

### Two live bugs fell out of the rewrite

Both would only have surfaced *after* an hour of GPU time.

1. **Region selection reached into the T-I evaluation block.** Filling a
   300-seed region at `gap`'s 6% hit rate needs ~5,000 eligible seeds, so
   selecting from `seed >= 2200` ran past 6000 and picked up **20 seeds** (17
   for `farb`) that are inside T-I's `[6000, 6300)`. The old `verify.py` would
   have caught it after the fact. It is now impossible by construction: every
   evaluation draws from one shrinking pool above `EVAL_BASE = 10000`, which
   sits above every block already measured.

2. **The index was too small to fill `farb`.** At 3.4% of eligible seeds, the
   8,000-seed index yielded 196 of the 300 needed. `seed_index.py` now reports
   per-mode availability and the index size each would require, for free, at the
   end of every index build.

A third, smaller one: `t2_common.py` imported numpy at module scope while its
docstring claimed the pure-CSV tools ran "on a laptop with no ManiSkill
install". They did not — the laptop has no numpy, and every analysis tool failed
to import. The module is now split into `geometry.py` (stdlib only) and
`harness.py` (sim only).

### The third mode and the third control were dropped

`nearbase` (`dist_min < 520 mm`) was a real finding and is kept in the record
above, but it is not carried into the deliverable. Two modes that fail at
different stages by different mechanisms is a stronger result than three slices,
and the budget is better spent on 300 fresh episodes each.

Dropping it also simplified `gap`. Its `dist_min >= 0.52` floor existed *only*
to keep it disjoint from `nearbase`; with that mode gone the floor did nothing
but dilute the effect:

| `gap` definition | n | success |
|---|--:|--:|
| `face_gap < 25`, `520 <= dist_min`, `dist_max < 760` (old) | 50 | 0.640 [0.501, 0.759] |
| `face_gap < 20`, `dist_max < 760` (**current**) | 44 | **0.523** [0.379, 0.662] |
| `face_gap < 20` raw, no control | 49 | 0.490 [0.356, 0.625] |

The old definition's interval overlaps the 0.713 baseline — it was reporting a
weaker effect than the data contains, because the floor removed the near-base
episodes that also fail. The current one keeps a single control, `dist_max <
760`, which exists to exclude **mode B's** factor and nothing else.

`farb` is unchanged. The two are now mutually exclusive on `dist_max < 0.76` vs
`dist_B >= 0.76` alone — verified at **0 overlap in 1,200 episodes**, asserted by
`t2/test_geometry.py`.

Recorded here so the decision is on the record rather than silently reversed by
someone who reads only the sections above.

### The pre-registered numbers, restated

These are what the 3 × 100 confirmation passes must reproduce on fresh seeds
drawn from above seed 10,000. They live in `geometry.DISCOVERY`, and
`test_geometry.py` asserts they still match `results/nominal.csv`.

| mode | n | success | 95% CI | grasped | placed | held \| placed |
|---|--:|--:|---|--:|--:|--:|
| nominal | 1200 | 0.713 | [0.687, 0.738] | 0.984 | 0.884 | 0.807 |
| `gap` | 44 | 0.523 | [0.379, 0.662] | 0.955 | **0.659** | 0.793 |
| `farb` | 41 | 0.561 | [0.410, 0.701] | **1.000** | 0.854 | **0.657** |

A confirmation landing outside its interval is informative, not a failure — the
prediction rests on ~40 episodes and the confirmation on 300. What must hold is
the **mechanism split**: `gap` low on placement with normal hold-given-placement,
`farb` at ~1.0 grasp and normal placement with hold-given-placement collapsed.
If that inverts, the two modes are not what this note says they are.

### What verification looks like now

Layered, because the layers fail for different reasons:

| | proves | needs |
|---|---|---|
| `test_geometry.py` | the *definition* of a failure mode is right | nothing |
| `test_verify.py` | **the checker catches things** — 11 corruptions, one at a time | nothing |
| `eval_modes.py`'s reset assertions | a bad episode is never *logged* | the sim |
| `verify.py` | the finished pass is what it claims | nothing |
| `policy_check.py` | the actions came from *these weights* | the sim |

`test_verify.py` is the one worth naming. A checker nobody has tried to fool is
not evidence, so it fabricates a valid pass and then corrupts it twelve ways —
including the two mistakes this note records above (evaluating on demonstration
seeds; a region pass sharing episodes with another block) and the misreading of
"3 seeds" from the section above. All twelve are caught.
