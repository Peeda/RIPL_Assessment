# CLAUDE.md

Context for coding agents working in this repo. Self-contained: everything an
agent needs to avoid re-making decisions that have already been paid for.

---

## The project

RIPL Lab assignment. Four tasks, each depending on the last:

- **T-I** — train a visual (RGB) diffusion policy on ManiSkill's `StackCube-v1`
  via imitation learning. Deliverable: success rate over 100 rollouts × 3 seeds.
  This is the baseline every later number is compared against.
- **T-II** — find 2 reproducible failure modes. Characterise them
  *quantitatively* as regions of the initial-state distribution (the two cube
  poses in SE(2), and above all their *relative* pose). Deliverable:
  per-failure-mode success rates + mp4s.
- **T-III** — LLM generates (a) a dense reward function and (b) an
  episode-configuration sampler biased toward the failure region.
- **T-IV** — `a = a_base + clip(Δ, −α, α)`. Base policy frozen; residual head
  trained with PPO against the T-III reward. Must show improvement on the
  targeted region **and** near-zero degradation on the nominal distribution.

The report is a running document, not a final step. Hyperparameters, VRAM
figures, wall-clock times and dead ends get pasted in as they happen. Debugging
narratives are explicitly requested by the assignment — they are deliverables,
not overhead. `notes/pusht-detour.md` is the first of them;
`notes/rgb-localisation.md` is the second and covers the T-I RGB arc;
`notes/t2-failure-modes.md` is the third and holds the T-II results.

**The task changed once already.** This repo was built against `PushT-v1` and
moved to `StackCube-v1` after the pipeline went green without ever producing a
policy with non-zero success. Everything measured during that period, and the
reasoning behind the decisions that did and did not survive the move, is in
`notes/pusht-detour.md`. Read it before re-deriving anything; do not import its
numbers into a StackCube table.

---

## Fixed decisions — do not relitigate

| Decision | Value | Why |
|---|---|---|
| Task | `StackCube-v1`, `motionplanning` demos | One raw `trajectory.h5`, a tuned DP *command* in `baselines.sh`, and — unlike `PushCube-v1` — not saturated. Note: **no published success numbers exist**, see below. |
| Control mode | `pd_ee_delta_pos` (**4-dim: 3 translation + gripper**) | Matches the published recipe. T-IV's α is only a coherent physical bound when what it bounds is a displacement in metres — see the residual row. |
| Replay flag | **`--use-first-env-state`** | Motionplanning demos are recorded under `pd_joint_pos`, so replay is a control-mode *conversion* and must simulate forward from the initial state. **This reverses the PushT rule**; see below. |
| Backend | `physx_cpu` everywhere | Matches the published recipe. Replay and eval backends must still match each other — the backend is in the output filename for that reason. |
| T-IV residual | 3 translation dims only; gripper passes through | See below. |
| gymnasium | **`==0.29.1`, pinned below 1.0** | 1.0 removed `final_info` from vector envs. The CPU baseline eval path reads it unconditionally, so `physx_cpu` + gymnasium≥1.0 = `KeyError` at the first eval. See traps. |
| Failure filtering | never `--allow-failure` | Training IL on failed demonstrations is worse than training on fewer. |
| Eval seed floor | **every eval block ≥ seed 1000** | Demo seeds start at 0 and run consecutively, so low seeds are the training set. See traps. |
| Demos / steps / iters | 100 / 200 / **30k state, 100k rgb** | From `baselines.sh`. The two runs do NOT share `total_iters`. `--num-demos` is the one of these that is a free lever — see below. |

### The replay flag reverses — read this before "fixing" it back

The old rule was `--use-env-states`, never `--use-first-env-state`. That was
correct **for PushT's `rl` demos**, which already sit in the target control mode:
replay only attaches observations, so pinning the sim to the recorded state each
step is free accuracy, and open-loop replay lost ~60% of episodes to drift.

It is wrong here. StackCube's motionplanning demos are recorded under a different
controller, so replay genuinely converts `pd_joint_pos` → `pd_ee_delta_pos` and
has to simulate forward. Stated properly:

> Use `--use-env-states` when replay only attaches observations.
> Use `--use-first-env-state` when replay converts the controller.

### The gripper's ORIENTATION is frozen for the whole episode

`pd_ee_delta_pos` pads the action with three zeros and `compute_target_pose`
"keeps the current rotation" (`pd_ee_pose.py:86-99`), so the end effector's
orientation is whatever it was at reset, for all 200 steps. The policy cannot
rotate the wrist.

Cube yaw is uniform on [0, 2π) (`random_quaternions(bounds=(0, 2*pi), lock_x,
lock_y)`), so the gripper meets cubes at every misalignment with equal
probability, and a 40 mm cube gripped near 45 deg presents 40*sqrt(2) = 56.6 mm.

**This is why `separation` is the wrong T-II axis and `face_gap` is the right
one** - see the T-II section below. It is also a report sentence in its own
right: the task hands the policy a fixed approach angle and a uniformly random
cube yaw, so part of the failure rate is structural.

### The T-IV residual must not touch the gripper dimension

The Panda's `pd_ee_delta_pos` is 4-dim: three translation plus one gripper.
StackCube requires grasping, so a residual `a = a_base + clip(Δ, −α, α)` applied
to all four perturbs the gripper signal, and a small perturbation at the wrong
moment drops the cube — degrading the base policy through a channel unrelated to
the spatial correction the residual is meant to learn.

**The residual head outputs 3 dims, bounded by α, added to the translation
components only. The gripper action passes through from the base policy
untouched.** This makes α a clean bound in metres, which is a better version of
the argument than the original translation-only framing, and it is a report
sentence.

### Datasets

If a dataset's action dim doesn't match the env, the dataset is wrong. Fix the
replay `-c` flag and regenerate. Do **not** "fix" the eval to match. Under
`pd_ee_delta_pos` the right answer is **4**; a 3 means the control mode is not
what you think it is.

---

## The task, as read from source

Verified against a local ManiSkill checkout (`~/ripl/ManiSkill`, shallow clone at
`62ff3a5`). Re-read rather than trusting this if the pod's version differs.

**Demos.** `StackCube-v1` is in `download_demo.py`'s `DATASET_SOURCES`, so the
download is the expected path; the motion planner is only a fallback.

**Hyperparameters** — from `examples/baselines/diffusion_policy/baselines.sh`,
which is the tuned source, not the README (the README only shows PickCube):

| | state | rgb |
|---|---|---|
| `--num-demos` | 100 | 100 |
| `--max_episode_steps` | 200 | 200 |
| `--total_iters` | **30000** | **100000** |

**The two runs do not share `total_iters`.** RGB trains 3.3× longer. This is the
one number that would have been silently wrong.

`max_episode_steps 200` sits against a **registered default of 50** — the README's
rule is ~2× mean demonstration length, and motionplanning demos are slow.

**Robot** is `panda_wristcam` by default, so RGB observations carry a wrist camera
alongside the 128×128 base camera.

**Success is three-part**, from `StackCubeEnv.evaluate()`: cubeA on cubeB within
±5 mm in xy and z, cubeA static, **and cubeA not grasped** — the robot must let go.
So "stacked but still holding it" and "stacked but knocked over" are failures, and
they are distinguishable. See the logging schema below.

**A dense reward already exists** (`compute_dense_reward`, 8-stage: reach → grasp
→ place → ungrasp+static). T-III's LLM-generated reward has a real baseline to be
compared against rather than invented in a vacuum. Do not delete this observation;
it is a report paragraph.

### The initial-state distribution — T-II's whole substrate

From `_initialize_episode`:

```python
xy = torch.rand((b, 2)) * 0.2 - 0.1        # ONE shared offset, both cubes
region = [[-0.1, -0.2], [0.1, 0.2]]
radius = ||[0.02, 0.02]|| + 0.001          # = 29.3 mm
cubeA_xy = xy + sampler.sample(radius, 100)
cubeB_xy = xy + sampler.sample(radius, 100)
```

Three consequences, all of which matter:

1. **The shared `xy` offset cancels in the relative pose.** Separation and relative
   yaw are statistically independent of where the pair sits on the table. The
   failure axis is therefore genuinely 2-D and not confounded with reach distance —
   a much cleaner T-II story than PushT would have given.
2. **`UniformPlacementSampler` rejection-samples** at `fixtures_radii + radius`, so
   the enforced minimum centre separation is **58.6 mm** for **40 mm** cubes. That
   only excludes physical overlap; it does *not* exclude the interesting regime.
3. **The failure region is inside the nominal support.** Simulating the sampler
   directly (200k draws):

   | | separation |
   |---|---|
   | enforced minimum | 58.6 mm |
   | p1 / p5 / p10 | 61 / 71 / 82 mm |
   | median | 162 mm |
   | max | 437 mm |

   `P(sep < 80 mm) ≈ 9%`, `P(sep < 70 mm) ≈ 4.6%`. With cubes 40 mm wide, a
   sub-80 mm separation leaves under 40 mm of clear space between faces, which is
   about what the Panda's gripper needs to descend without fouling the other cube.

   **This is the T-II hypothesis to test first**, and it is testable on nominal
   rollouts — no sampler override needed to *find* the region, only to *over-sample*
   it, which is exactly T-III's job.

Treat the 9% as a prior, not a result. It says the region is reachable at n=100
(≈9 episodes), which is too few for a per-region success rate — hence T-II's
requirement to resample fresh episodes from the region rather than reusing the
rollouts that identified it.

### Still open

- **Eval cost.** The "100 envs is essentially free" result in
  `notes/pusht-detour.md` was measured on `physx_cuda` and **does not apply
  here at all**: `physx_cpu` raises `RuntimeError` for `num_envs > 1` and
  vectorises by *subprocess* instead (`gym.vector.AsyncVectorEnv`, forkserver,
  one process per env, via `make_eval_envs`). So eval cost scales with CPU
  cores, not with batch width, and `--num_eval_envs` is a process count. Leave
  it at train.py's default of 10 unless the pod has cores to spare. Smoke gate 4
  prints single-env ms/step; divide by the process count.
- **Where the base policy actually lands.** Needs to be above 0 and below 1. See
  Current state.

---

## There is no published DP success number to compare against

`docs/source/user_guide/learning_from_demos/baselines.md` lists Diffusion
Policy with **Results: WIP**. Same for BC, ACT, RFCL, RLPD. So `baselines.sh`
is a set of *tuned commands*, not a set of *verified results*, and there is no
maintainer figure for StackCube DP — state or rgb.

This matters more than it sounds, because "stay on documented hyperparameters"
was justified by comparability to published numbers. Those numbers do not
exist. What the recipe still buys is a sane starting point that someone tuned;
what it does not buy is a target. Concretely:

- **A low number cannot be called a bug by comparison.** It can only be judged
  against what the task needs, which here is T-II's requirement: meaningfully
  above 0 and meaningfully below 1.
- **`--num-demos 100` is a free lever.** The replay produces ~990 trajectories
  and the recipe uses 100 of them; the rest are already on disk. For a state
  policy 100 is plainly enough. A visual policy also has to learn perception
  from roughly 100 × ~110 ≈ 11k frames, which is not obviously enough. Raising
  it is cheap, does not touch the LR schedule, and costs only dataloading and
  memory.
- Record the demo count with every number, since it is now a variable.

---

## `total_iters` is an LR hyperparameter, not a budget

`train.py` uses a diffusers cosine schedule with 500 steps of linear warmup,
stepped every batch (`train.py:343-347`, `:414`):

```python
get_scheduler(name='cosine', num_warmup_steps=500,
              num_training_steps=args.total_iters)
```

So `--lr 1e-4` is the **peak**, not a constant: 0 → 1e-4 over 500 iters, then
cosine decay to **0** at `total_iters`.

| iter | state (30k) | rgb (100k) |
|---|---|---|
| 500 | 1.00e-4 | 1.00e-4 |
| 15,000 | 5.13e-5 | 9.49e-5 |
| 30,000 | 0 | 7.98e-5 |

Because the schedule is tied to `total_iters`:

- **Early-stopping a long run ≠ a short run.** Killing the 100k rgb run at 30k
  leaves the weights at 8e-5, 80% of peak, with none of the low-LR annealing
  that usually gives the last quality gain. Not comparable to a 30k-scheduled
  run, and not the published recipe.
- **A run cannot be extended** by raising `total_iters` — that resumes under a
  different LR trajectory.
- **The state and rgb runs are on deliberately different schedules.** That comes
  from `baselines.sh`; it is a second reason those two numbers stay separate
  variables in `t1/run_pipeline.sh`.

The PushT-era instinct "set iters high, watch, stop when flat" is therefore
**wrong here**. If a run is too long, shorten `total_iters` and rerun from
scratch — do not truncate. A flat success curve mid-run is expected while the
LR is still high; judge convergence at the schedule's end, not from the plateau.

---

## What the T-I number actually is

`train.py` logs eval metrics through a tensorboard `SummaryWriter` with
`wandb.init(sync_tensorboard=True)`, so in wandb they appear namespaced as
**`eval/<key>`**. There is no metric called "success rate". The keys, from
`CPUGymWrapper` (`mani_skill/utils/wrappers/gymnasium.py:59-83`):

| wandb key | meaning |
|---|---|
| `eval/success_once` | task succeeded at **any** step of the episode |
| `eval/success_at_end` | task was in a success state at the **final** step |
| `eval/fail_once`, `eval/fail_at_end` | same, for the failure criterion |
| `eval/return`, `eval/reward`, `eval/episode_len` | sparse-reward sums |

`success_at_end` exists only because `make_eval_envs` sets
`ignore_terminations=True`, which runs every episode to the full horizon
instead of stopping at first success.

**Decide once: `success_once` is the reported T-I number.** It is what
ManiSkill's published baselines report, so it is the only one comparable to
them, and T-IV's before/after must use the same key.

**But record both, and say so in the report**, because on StackCube they can
genuinely differ. Success requires cubeA on cubeB *and* static *and* released —
a cube that is stacked at step 120 and topples by step 200 scores
`success_once=1`, `success_at_end=0`. A large gap between the two is not noise;
it is a real failure mode (unstable placement), and it is a T-II candidate in
its own right, distinct from the separation-based one.

**Eval cadence.** `--eval_freq` defaults to 5000 and `baselines.sh` does not
override it, so 30k iters gives ~7 points. That is a coarse curve for a T-I
plot. Lowering it to 1000 does not change training — it costs wall clock and
gives `save_on_best_metrics` more chances to catch a good checkpoint — so it is
a safe deviation if the curve matters. `--num_eval_episodes` is 100 by default,
which is the assignment's rollout count.

**If a metric is missing from wandb, check the stdout log first.**
`evaluate_and_save_best` prints every key it computes. If `success_once:` lines
are in the log, evaluation is fine and the question is wandb sync or the key
name; if they are absent, evaluation is not running.

---

## Measured: state works, RGB does not localise (100 demos)

Both from `rollout_log.py` against `best_eval_success_once.pt`, same replay,
same actions, same control mode — the runs differ only in observation mode.

> **The `success_once` row below is mislabelled and is really `success_at_end`.**
> `rollout_log.py` read the top-level `info['success']` at the final step;
> `CPUGymWrapper(record_metrics=True)` nests the real `success_once` under
> `info['episode']`, and `ignore_terminations=True` means every episode runs to
> the full horizon, so the top-level key is the end-of-episode value. Fixed —
> the harness now logs both columns — but these two numbers predate the fix.
> Since `success_once >= success_at_end` always, the true `success_once` is at
> least 0.680 / 0.040. Re-measure before either goes in the report, and never
> compare a pre-fix rollout number against wandb's `eval/success_once`.

| | state | rgb |
|---|---|---|
| `success_once` | **0.680** | **0.040** |
| mean reach error to cubeA | **4 mm** | **86 mm** |
| corr(cubeA_x, tcp_x) | +0.999 | +0.689 |
| corr(cubeA_y, tcp_y) | +1.000 | +0.819 |
| reach-spread / cube-spread | 1.00× | 0.79× (x), 0.84× (y) |

"Reach" is TCP xy at deepest descent — where the policy decided to grasp.

**The RGB failure is localisation, and it is scatter rather than bias.**
Decomposing the error, shrinkage toward the mean is mild (0.79–0.84×) while the
error standard deviation is ~109 mm combined. The policy aims in roughly the
right direction and lands most of a cube-width away. Against a 40 mm cube that
is a miss. It is **not** blind — correlation of 0.69–0.82 is real visual
tracking — so "ignores the image" is the wrong diagnosis and was one this repo
briefly held.

**The state policy at 4 mm is effectively a perfect localiser and still fails
32% of the time.** That is the more interesting number. Those failures happen
*downstream of reaching* — grasp, place, release, settle — which is exactly
what the three partial-success flags in the logging schema separate. **T-II has
usable material on the state policy today.**

**0.680 also clears the T-I gate**: meaningfully above 0, meaningfully below 1.
StackCube was the right task choice; the open question is only whether the
visual policy can be brought to the same place.

### What this does and does not license

- Reach error is a better instrument than success rate for debugging a visual
  policy: it is continuous, it has a physical scale (mm against a 40 mm cube),
  and it separates a perception failure from a manipulation failure. Report it.
- **Correlation alone is misleading here.** r = 0.82 coexists with an 86 mm
  error, because r is invariant to scale and offset. Always quote the error in
  mm alongside it.
- Reach error and `success_once` are separate instruments and both are needed;
  see the 800-demo result immediately below.

### 800 demos, pooled encoder: `success_once` ≈ 0.43 (recipe otherwise verbatim)

`baselines.sh`'s StackCube rgb line with `--num-demos 800` instead of 100,
everything else identical. Result: **more demos help a lot, and then stop.**

| demos | encoder | `success_once` |
|---|---|---|
| 100 | pooled (upstream) | 0.040 |
| 800 | pooled (upstream) | **~0.43** (final eval 0.43, best 0.49) |

**The memorisation hypothesis was right** — 0.04 → 0.43 is a real result, no
longer a guess. Record it; it is a report paragraph on its own.

**But the run converged.** ~21 eval points at `eval_freq=5000` over 100k iters,
i.e. the schedule completed. Flat from step 60k: the last 40k iterations
oscillate 0.42–0.49 with no trend, and at `--num_eval_episodes 100` the
Bernoulli SE is ≈5%, so that entire oscillation is one standard error. At 800
demos (~88k transitions, `batch_size` 256) 100k iters is ~290 epochs.

**So more iterations will not help, and neither will more demos** — 800 is
already most of what the replay produces. The remaining gap to the state
policy's 0.680 is representational, which is what the next section is about.
Do not spend another 100k-iter run on a data or budget lever.

---

## The RGB encoder throws away position — `patches/0001`

*Narrative version, with the two hypotheses that were wrong first:*
`notes/rgb-localisation.md`. This section stays the canonical record of the
numbers; the note tells the story around them. Do not maintain both.

**This is the reason the visual policy plateaus.** It is a property of
ManiSkill's baseline, not of anything this repo did.

`train_rgbd.py:272-274` hardwires `PlainConv(..., pool_feature_map=True)`, and
that flag selects `nn.AdaptiveMaxPool2d((1,1))` (`plain_conv.py:56-58`). A
128×128 image goes through four max-pools to an **8×8×128** feature map, and
the whole 8×8 grid is then collapsed to **one number per channel** before the
FC to 256. There are no spatial coordinates left in the representation.
Position survives only as "which of 128 channels fired" — a coarse code with no
spatial support.

That is exactly the measured signature: r = 0.69–0.82 (real tracking, not
blind) with 86 mm mean reach error and σ_err ≈ 109 mm against a **40 mm** cube.

It binds on StackCube specifically because **under `--obs-mode rgb` the state
vector carries no cube pose.** `stack_cube.py:133-143`:

```python
obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
if "state" in self.obs_mode:          # <- not taken under rgb
    obs.update(cubeA_pose=..., cubeB_pose=..., tcp_to_cubeA_pos=..., ...)
```

So the RGB policy sees `tcp_pose` + `qpos/qvel`, and 100% of cube localisation
must flow through the pooled encoder. The state policy reaches to 4 mm because
it is handed the pose; the gap between 0.68 and 0.43 is not "vision is harder",
it is that the single quantity vision must supply is the one the encoder discards.

**`patches/0001` makes it an `Args` field defaulting to `False`** (spatial map
kept), rather than flipping the constant, so both arms stay runnable and the
value reaches wandb's config through `vars(args)` (`train_rgbd.py:453`).

- **Checkpoints are not interchangeable across this flag.** `visual_encoder.fc.0.weight`
  is `(256, 128)` pooled and `(256, 8192)` spatial. `rollout_log.py` infers which
  from the weights and configures `Args` to match — do not add a manual flag for it.
- **`128*4*4*4` in the non-pooled branch is already correct**, since it equals
  `128*8*8 = 8192` for a 128×128 input. Only the inline `[4,4]` shape comments
  are stale. Do not "fix" the arithmetic.
- **`self.aug` is dead code.** `train_rgbd.py:301` guards it with
  `hasattr(self, "aug")` and nothing ever assigns it, so there is **no image
  augmentation** anywhere in this baseline. That is the obvious second lever if
  the encoder change is not enough; it is not part of `patches/0001`.

### Deviating here costs nothing in comparability

The usual objection — "stay on documented hyperparameters so the number is
comparable" — does not apply, because there is no published DP number for
StackCube to be comparable to (see above; `baselines.md` lists DP as WIP). The
recipe buys a sane starting point, not a target. **Report the deviation, the
before/after, and the mechanism.** A measured architectural fix with an A/B is
a stronger T-I section than a recipe reproduced without understanding.

## T-II logging schema

Decide this once, here, and never re-derive it. Angle-wrap conventions are what
make a failure characterisation incoherent.

Per episode, log:

```
{seed, env_idx, cubeA_x, cubeA_y, cubeA_theta,
                cubeB_x, cubeB_y, cubeB_theta,
 separation, relative_yaw, ep_len,
 success, is_cubeA_grasped, is_cubeA_on_cubeB, is_cubeA_static}
```

- `separation` is XY Euclidean distance between cube centres (metres).
- `relative_yaw` is `cubeB_theta − cubeA_theta`.
- **All angles wrapped to `(−π, π]`.** Both the absolute thetas and the relative
  yaw. Wrap once at log time, never at analysis time.
- The last three flags come free from `evaluate()`'s returned info dict. **Log
  them.** They turn a binary failure into a three-way taxonomy at zero cost:
  never grasped / grasped but misplaced / placed but not released or not settled.
  Two "failure modes" that differ in which flag is false are genuinely different
  modes, and that distinction is most of what T-II is being graded on.

`separation` and `relative_yaw` were the *hypothesised* failure axes, and both
have since been superseded by measurement — see Current state. Keep logging them
(they are pinned columns and the report shows the before/after), but regress
against the derived features in `t2_common.geom_features`:

- **`face_gap`** = `separation − extentA(bearing) − extentB(bearing)`, the
  clearance between cube *faces*. Both yaws enter, through the A→B bearing. This
  is why `relative_yaw` alone measures nothing: yaw is not a scalar you can
  regress against without a direction to resolve it along.
- **`dist_A` / `dist_B` / `dist_max` / `dist_min`** from `PANDA_BASE_XY`, not the
  world origin. Success against `dist_max` is an inverted U, so **bin both
  tails**; fitting a trend hides half the effect.

These are derived, not logged, so every CSV mined before they existed gains them
at analysis time via `geom_from_row`. Nothing needs re-mining.

**Log full initial states + success flags for every rollout, always** — including
diagnostic runs. T-II is near-impossible to do well retroactively without this.

**T-II requires resampling *fresh* episodes** from the identified failure region
and re-measuring there. Reporting the failure rate on the same rollouts used to
find the region measures noise, not a failure mode.

Bernoulli SE is `sqrt(p(1−p)/n)` ≈ 4–5% at n=100. "Near-zero degradation" is a
claim that 3 seeds × 100 rollouts supports only weakly. Say so in the report.

---

## Infrastructure

RunPod, **no network volume, no object storage**. Every pod is a clean rebuild.

- Files move by `rsync` over **direct TCP SSH**, requiring a public IP with
  exposed port 22. The `ssh.runpod.io` proxy cannot do SCP/SFTP/rsync and the
  error doesn't say so. Run `bash setup/transfer.sh check` first thing on every pod.
- **Stop ≠ Terminate.** A stopped pod's disk survives and is reachable by
  restarting with zero GPUs. A terminated pod is gone. Never terminate a pod you
  haven't pulled from. Running out of credit auto-stops a pod; drained accounts
  lose the disk after ~48h.
- **`SYS_PTRACE` is not settable on RunPod**, so `py-spy` may not work — Docker's
  seccomp profile blocks `process_vm_readv`. For a hang, add to the target
  script:
  `import faulthandler, signal; faulthandler.register(signal.SIGUSR1, all_threads=True)`
  then `kill -USR1 <pid>` twice a few seconds apart. Identical stacks = stuck,
  different = slow. Note `-X faulthandler` does **not** catch SIGQUIT.
- Pin the GPU model across pods, **and record it with every measurement.** T-I
  reports training time and VRAM; T-IV reports wall-clock. Numbers from different
  cards don't form a table, and the PushT figures are missing that label. Avoid
  RTX 5090 / Blackwell (open ManiSkill torch-version issue).
- Container disk ≥ 60 GB. `NVIDIA_DRIVER_CAPABILITIES=all` at deploy time.

Paths (`env.sh` must be sourced in every new shell — interactive shells on
RunPod do not inherit the container environment):

```
$RIPL_ROOT        /workspace/ripl
$MANISKILL_REPO   $RIPL_ROOT/ManiSkill
$MS_ASSET_DIR     $RIPL_ROOT/maniskill_data
demos             $MS_ASSET_DIR/demos/StackCube-v1/motionplanning
diffusion policy  $MANISKILL_REPO/examples/baselines/diffusion_policy
this repo         ~/RIPL_Assessment   (note: outside /workspace)

On the laptop, for reading source without a pod:
~/ripl/ManiSkill    shallow clone, no install needed
~/ripl/demos        rsync target for pulled datasets
```

Inside this repo, one directory per assignment task (see The scripts, below):

```
setup/    pod build, patching, transfer, smoke gates
t1/       run_pipeline.sh - data replay + both training runs
t2/       run_t2.sh and the mining/analysis harness
patches/  patches applied to the ManiSkill checkout
notes/    debugging narratives; report material, not overhead
figures/  analyze_rollouts.py's output, committed
```

---

## The scripts

**One directory per assignment task**, plus `setup/` for everything that is
task-independent. `t3/` and `t4/` will sit alongside `t1/` and `t2/`; the
numbering means the same thing in all four.

### `setup/` — pod infrastructure, no task attached

| | |
|---|---|
| `setup_runpod.sh` | Builds a fresh pod: packages, Vulkan/EGL ICDs, venv, ManiSkill clone. Writes `env.sh`. Deliberately checks nothing at runtime. |
| `apply_patches.sh` | Applies `patches/*.patch` to the ManiSkill checkout. `apply` \| `status` \| `revert`. Idempotent. Run after every `setup_runpod.sh`. |
| `smoke_test.sh` | Five gates, ~3 min. Sim constructs; renderer emits real pixels; dataset action dim matches; seeds determine initial states; the real evaluate() runs. Gate 4 is T-II's prerequisite. |
| `transfer.sh` | `check` \| `info` \| `send`. Run `check` before anything else on a new pod. |

### `t1/` — data and training

| | |
|---|---|
| `run_pipeline.sh` | The recipe, plus one deviation (`POOL_FEATURE_MAP`). `data` \| `train` \| `train-rgb` \| `all`. |

Thin on purpose. It is one command because StackCube ships one raw
`trajectory.h5`; the T-I *numbers* come out of `t2/`, see below.

### `t2/` — the failure-mode harness

| | |
|---|---|
| `run_t2.sh` | Driver. `index` \| `t1` \| `mine` \| `region` \| `backend` \| `videos` \| `analyze` \| `plan` \| `all`. |
| `run_all.sh` | **Everything T-II still needs, in order.** `preflight` \| `index` \| `size` \| `modes` \| `feasible` \| `backend` \| `analyze` \| `verify` \| `all`. Derives `CKPT`; resumable. |
| `run_modes.sh` | The three pre-registered per-mode evaluations. `size` \| `gap` \| `farb` \| `nearbase` \| `analyze` \| `all`. |
| `verify.py` | Asserts a finished pass is what it claims. No sim. Exits non-zero, so it can gate a report. |
| `t2_common.py` | Angle conventions, cube + **geometry** features, Wilson, checkpoint loading, manifests. |
| `seed_index.py` | seed → initial state, **policy-free**. No GPU, minutes. |
| `select_seeds.py` | Pick seeds by any condition on the initial state. `--dry-run` sizes a pass. |
| `mine_rollouts.py` | The overnight miner. Parallel, resumable, fully logged. |
| `record_seeds.py` | mp4s for named seeds, retried until the outcome matches. |
| `analyze_rollouts.py` | Curves, taxonomy, Wilson intervals, video shortlist. |
| `rollout_log.py` | The localisation probe. `num_envs=1`; produced the reach-error numbers. |
| `backend_check.py` | Does `physx_cuda` reproduce `physx_cpu`? Paired by injected initial state, not by seed. |
| `reach_map.py` | Reachable set for the frozen top-down gripper. **Policy-free**, needs ManiSkill. |
| `demo_feasibility.py` | The motion planner as a feasibility oracle. A metadata join; no sim needed. |

Three placement calls, so they are not re-litigated:

- **`t2_common.py` imports gymnasium and torch *optionally*.** `analyze_rollouts.py`,
  `select_seeds.py` and `demo_feasibility.py` are pure-CSV tools and must run on
  a laptop with no ManiSkill install — which is the same reason the saved state
  arrays carry their own column names. Only the rollout half needs the sim, and
  only it fails without one.

- **`rollout_log.py` lives in `t2/` even though its reach-error numbers are T-I
  material.** It does `import t2_common`, and python only puts a script's *own*
  directory on `sys.path` — from `t1/` that import breaks. It is a rollout
  harness sharing the T-II conventions module, so it lives with them.
- **`run_t2.sh t1` is a T-I deliverable produced by the T-II harness**, and that
  is deliberate rather than a filing mistake. T-I's error bars cannot come from
  the training logs (see above), so the standalone harness is what produces
  them. One tool, two deliverables.

`ENV_ID` is a variable in both `t1/run_pipeline.sh` and `setup/smoke_test.sh`.
The task has changed once; changing it again should cost one export, not a
rename.

`figures/` at the repo root is `analyze_rollouts.py`'s output and is committed.

---

## Changing the pod's python environment

`setup/setup_runpod.sh` is **safe to rerun** and does apply dependency changes — the
`uv pip install` block is unguarded, so it runs every time, and an exact pin
(`gymnasium==0.29.1`) downgrades an already-installed newer version. What the
guards protect is only creation: `[ -d "$VENV" ]` skips venv creation, and
`[ -d "$ROOT/ManiSkill" ]` skips the clone — so **rerunning never updates the
ManiSkill checkout**. To move that, `git pull` it by hand.

But for a one-package change, rerunning means an `apt-get update`, a torch
re-resolve and a full Vulkan gate for no reason. Prefer the targeted install:

```bash
source /workspace/ripl/env.sh
uv pip install --python /workspace/ripl/venv/bin/python "gymnasium==0.29.1"
python -c "import gymnasium; print(gymnasium.__version__)"
bash setup/smoke_test.sh    # gate 5 is the one that proves it
```

Rerun the whole script when the *machine* changed (new pod, new container), not
when one dependency did.

Note the unpinned `torch torchvision` line: `uv pip install` without
`--upgrade` treats an already-satisfied requirement as a no-op, so a rerun does
not silently move torch. Do not add `--upgrade` to it — the GPU/torch pairing
is load-bearing and Blackwell cards already have an open ManiSkill issue.

---

## Working rules for agents

**Git**
- The repo is the source of truth. Commit hygiene is an explicit evaluation
  criterion — small, coherent commits with real messages.
- **Commit from the laptop only.** The pod clones and `git pull`s; it never
  pushes. Anything uncommitted on a pod is one Terminate away from gone.
- **Nothing large goes into git.** `.gitignore` covers `*.h5`, `runs/`,
  `wandb/`, `env.sh`. Committing a big blob is one of the few mistakes with no
  cheap undo — `git rm` doesn't remove it from history. Never `git add -A`
  without checking what it caught.

**Running things**
- Anything long-running goes in `tmux`, then poll the log. Do not foreground a
  multi-hour training run.
- Never delete or overwrite a dataset without asking.
- When capturing terminal output for later analysis, redirect to a file
  (`2>&1 | tee log`) rather than `tmux capture-pane`. Capture interleaves
  redrawn progress bars and destroys exactly the timing information you wanted.
- **Sanity-check any throughput number against a stopwatch before planning on
  it.** A script printing `it/s` looks authoritative and was wrong by 10× on
  PushT. `notes/pusht-detour.md` lists the four ways it went wrong.

**Data hygiene for the report**
- `--track` (wandb) from the first real training run. It captures loss curves,
  `torch.cuda.max_memory_allocated()` and step timings off-pod automatically —
  all T-I and T-IV deliverables. Requires `wandb login` first or it fails at
  startup; `t1/run_pipeline.sh` checks.
- At high `--num_eval_envs` the logged scalar is a single mean over the batch, so
  per-eval variance is not recoverable from wandb. **T-I error bars must come
  from a standalone harness, not the training logs.**

---

## Known traps

- **`ConnectionResetError: [Errno 104] Connection reset by peer` from an
  `AsyncVectorEnv` is a dead worker, not a network problem.** The most likely
  cause in this repo is running the parent from a heredoc: `multiprocessing`'s
  `forkserver` (and `spawn`) re-import `__main__` in every child, so `__main__`
  must be importable from a real path. Under `python - <<'PY'` it is not, the
  child dies during the handshake, and the parent reports a reset socket —
  which reads like a pod fault and is not one. Anything touching
  `AsyncVectorEnv` must be **written to a file and run**, with the work under
  `if __name__ == '__main__':` and imports at module level so the child's
  re-import restores `sys.path`. `train.py` is a real file, which is why it is
  unaffected; smoke gate 5 writes itself to `$RIPL_ROOT/smoke/gate5_eval.py`.
- **`KeyError: 'final_info'` at the first eval means gymnasium ≥ 1.0.** This
  bit us for real. Gymnasium 1.0 removed `final_info`/`final_observation` from
  vector envs and switched autoreset to `NEXT_STEP`. `ManiSkillVectorEnv`
  synthesises `final_info` itself (`vector/wrappers/gymnasium.py:167`) so
  `physx_cuda` is immune — but the CPU path uses stock
  `gym.vector.AsyncVectorEnv`, and every baseline's `evaluate()` reads
  `info["final_info"]` with no guard. So the bug is invisible on GPU and fatal
  on CPU, which is the backend this pipeline uses throughout. ManiSkill declares
  `gymnasium>=0.29.1` with no upper bound and branches on `IS_GYMNASIUM_1`, so
  0.29.1 is supported, not a downgrade hack. `setup/setup_runpod.sh` pins it; smoke
  gate 5 catches it in seconds by running the real `evaluate()`.
- **`physx_cpu` raises `RuntimeError` for `num_envs > 1`.** It vectorises by
  subprocess, not by batching, so any `gym.make(..., num_envs=N,
  sim_backend='physx_cpu')` with N>1 dies immediately. Use
  `gym.vector.AsyncVectorEnv` (or the baseline's `make_eval_envs`, which does).
  Every habit carried over from the `physx_cuda` era violates this.
- **A seed does not address an episode on `physx_cuda`, and the assert that
  should catch that PASSES AND LIES.** `_initialize_episode` draws cube poses
  with `torch.rand` under `with torch.device(self.device)`
  (`stack_cube.py:78-100`), so the CUDA generator gives different states than
  the CPU one for the same seed. Worse, `reset()` seeds the whole batch from
  `self._episode_seed[0]` (`sapien_env.py:950-953`) while `_initialize_episode`
  draws `(b, 2)` at once - so at width > 1 every env's state derives from env
  0's seed and depends on the batch width. `mine_rollouts.py` asserts the
  episode seed came back as requested; `_episode_seed` IS recorded per-env, so
  the assert passes while the states it is supposed to guarantee are not the
  ones asked for. **The seed index and every T-II seed list are `physx_cpu`
  artifacts.** Do not port them. `t2/backend_check.py` compares the backends by
  replaying exact initial states instead, which is what licenses T-IV training
  on GPU and evaluating on CPU.
- **`train.py`'s flag names move between ManiSkill releases.** Re-check
  `--help` rather than trusting any doc, including this one.
- **`--capture-video` defaults to True.** Pass `--no-capture-video` for anything
  you are timing.
- **`train.py` hard-asserts if `--num-demos` exceeds what's in the file.** Derive
  the count from the h5 rather than assuming the replay kept everything.
- **`evaluate()`'s episode count is not what it looks like.** The loop is sized by
  `--num-eval-episodes`, not `--num_eval_envs`; the latter only sets batch width,
  and the `Evaluated N episodes` print counts *batches*. Looks like a bug, isn't.
- **`IterationBasedBatchSampler` stalls on very small datasets.** A hang at
  iteration 0 usually means the dataset is tiny, which usually means the replay
  went wrong.
- **Motionplanning demos are planner output, not human teleop.** Close to
  unimodal per initial condition, which mutes diffusion policy's usual edge over
  plain BC. Belongs in the T-I discussion.
- **Demo episode seeds start at 0 and run consecutively, so low seeds are the
  TRAINING SET.** `examples/motionplanning/panda/run.py:44-101` does
  `seed = start_seed` (0), `solve(env, seed=seed)`, `seed += 1` per attempt.
  ~990 trajectories were replayed and 800 trained on, so roughly **seeds
  [0, 1000) are training initial states**. Evaluating there measures
  memorisation. Measured on the same checkpoint: **0.910 on seeds 0-299 vs
  0.713 on held-out seeds 1000-2199**, with the gap uniform across every
  separation bin, flat over the run and flat across workers. `t2/run_t2.sh`
  now refuses `T1_BASE < DEMO_SEED_CEILING`. This bit us for real — the first
  production T-I pass ran on seeds 0-299.

---

## Current state

*Volatile — update as work lands.*

- Repo retargeted at StackCube; PushT scripts deleted, its record kept in
  `notes/pusht-detour.md`. No PushT number belongs in a StackCube table.
- **Data and both policies exist.** Demos replayed (state + rgb, ~990 trajs).
  State: `success_once` **0.680**, 4 mm reach error, 100 demos. RGB: **0.040**
  at 100 demos, **~0.43** at 800 demos — both on upstream's pooled encoder.
- **Diagnosed:** the RGB plateau is the encoder's global max-pool discarding
  spatial layout, not data volume and not training budget. See the two sections
  above. `patches/0001` turns it off; `t1/run_pipeline.sh` defaults to the spatial
  encoder and records the variant in the run name.
- Hyperparameters, demo availability and the initial-state distribution are all
  verified from source. A shallow ManiSkill checkout lives at `~/ripl/ManiSkill`
  on the laptop for exactly this — read source locally instead of burning pod
  time on it. Keep it **stock**: `bash setup/apply_patches.sh revert` after any local test,
  or local source-reading stops matching upstream.
- **The spatial-encoder RGB checkpoint exists** (800 demos, 100k schedule,
  `patches/0001`). It is the T-I deliverable and T-IV's frozen base. The full
  arc that produced it — the wrong "it's blind" diagnosis, the data lever and
  where it ran out, the encoder finding — is written up in
  `notes/rgb-localisation.md`.
- **The production T-II pass has run**, and its axes have since been corrected.
  1,900 episodes, RTX 4090, 55 min at ~2,000 ep/h. Full write-up in
  `notes/t2-failure-modes.md`. Headlines:
  - **T-I = 0.737**, SD 0.078 over 3 × 100 on held-out seeds 6000–6299; 0.713
    [0.687, 0.738] on the 1,200-episode nominal pass. The 0.910 from the first
    `t1` pass was measured on seeds 0–299, which are *training* initial states —
    see the demo-seed trap. That gap is a memorisation result, not a T-I number.
  - **`separation` is the wrong axis, and both original modes are restated on
    better ones.** Measured on the same 1,200 held-out episodes, no new data:
    - **`face_gap`** — the clearance between cube *faces* along the A→B bearing,
      so both yaws enter. 0.500 at 5–20 mm rising to 0.794 at 70–120 mm, and it
      survives controlling for separation (within sep < 100 mm: 0.524 low-gap vs
      0.670 high-gap). Mechanism in the source: the gripper's orientation is
      frozen at reset, so the policy cannot square up to a misaligned cube.
    - **`dist_max` / `dist_min` from `PANDA_BASE_XY`** — an inverted U. 0.467
      below 520 mm, 0.774 mid-band, 0.508 above 760 mm. The old "separation >
      260 mm" mode is largely this one seen through a correlated variable; the
      2×2 puts the separation tail at −0.084, the reach tail at −0.198, both at
      −0.356.
  - **The reach band's two ends fail by different mechanisms**, which is what
    makes them separate modes. Far cubeA spikes `never grasped` to 0.149 against
    a 0.016 base rate; far cubeB leaves it at 0.015 and fails at placement;
    near-base grasps normally and fails at placement at 0.178 vs 0.069.
  - **Toppling is not a third mode.** It is 28.7% of placements, but no
    initial-state feature predicts it. What does is `cubeB_displacement` (hold
    rate 0.919 → 0.032), an outcome — so it is the face-gap mode's downstream
    consequence. This closes a question `CLAUDE.md` previously left open.
  - **T-IV carries two targets** — `face_gap < 25 mm` and far-cubeB-with-cubeA-
    comfortable — chosen because they fail at *different stages* by *different*
    mechanisms. See the decision point below; an earlier claim that the reach
    mode was not residual-fixable has been retracted there.
  - **Beware `far_is_B` as a mechanism split.** The `never_grasped = 0.149` for
    "far cube = A" mixes in both-cubes-far episodes. Isolating cubeA ≥ 760 mm
    with cubeB < 720 mm gives 0.083, and the grasp failures actually concentrate
    in the both-far corner (0.815 above 740 mm). Condition on the *other* cube
    when attributing a mechanism to one of them.
  - The gate holds comfortably: 0.713 is well below 1, and the nominal pass
    yields 344 failures rather than the ~13 that 0.87 would have given.
- **Next, in order:**
  1. `bash setup/apply_patches.sh` on the pod (after `setup_runpod.sh`, which
     re-clones ManiSkill and wipes the patch).
  2. `bash t2/run_all.sh` — everything below in one command, resumable, ~1 h.
     The stages are listed separately here because they can be run piecemeal.
  3. `python t2/demo_feasibility.py $MS_ASSET_DIR/demos/StackCube-v1/motionplanning/trajectory.json`
     and `python t2/reach_map.py` — do the far-reach failures belong to the
     policy or to the task? Both are cheap; the first is a metadata join.
  4. `bash t2/run_t2.sh backend` — needs a GPU. Licenses T-IV's train-on-GPU,
     evaluate-on-CPU split. Not part of `all`.
  5. `bash t2/run_t2.sh analyze`, then `videos` on the shortlist it prints —
     preferably seeded from the region lists so each clip illustrates a mode.
  6. Pull before stopping the pod: `bash setup/transfer.sh info`. Figures do
     **not** come back by git; the pod never pushes.

---

## The T-II harness

`bash t2/run_t2.sh <stage>` — the file-by-file map is in The scripts, above.
Read this section before adding a stage.

The rollout stages use **disjoint seed blocks**: `t1` takes [0, 300) as three
blocks of 100 with three policy seeds, `mine` takes [1000, 1000+NOMINAL), and
`region` draws only from seeds above that. Disjointness is not tidiness — a T-I
number that shares episodes with the T-II substrate is not an independent check
on it, and re-measuring a region on the rollouts that found it measures noise.

`rollout_log.py` stays what it is: the **localisation probe**, `num_envs=1`, the
thing that produced the reach-error numbers above. It shares `t2_common.py` with
the miner so the two cannot disagree on the wrap convention or on which weights
get loaded.

### The seed index is the load-bearing idea

`reset(seed=s)` is deterministic (`sapien_env.py:950-953`), so **a seed is a
lossless 8-byte encoding of the whole 70-float initial state.** Tabulating
seed → cube poses needs resets only — no policy, no 200-step rollout, no GPU.

That turns T-II's "resample *fresh* episodes from the failure region" into
rejection sampling over integers: filter the index for `separation < 0.080`,
hand the surviving seeds to the miner. The episodes that come back are drawn
from exactly the env's own conditional distribution given the region. No state
injection, no distribution shift, every episode reproducible from one integer.

It also means **no exploratory question about initial states can be foreclosed
by a logging decision.** Anything not logged is a reset away from being
recovered.

The alternative — injecting poses via `options={"reset_to_env_states": ...}` —
replaces `_initialize_episode` wholesale (robot qpos and table pose would have
to be synthesised too) and broadcasts identically to every worker under
`AsyncVectorEnv`. Save it for T-III, where a biased distribution is the goal.

### What gets logged, and why that split

**Initial state: the seed suffices, but log it anyway.** Redundant by
construction, ~1.7 MB, and it makes the dataset self-contained — the analysis
reproduces from committed files without a ManiSkill install. It also carries
the one part of the initial state the cube columns ignore: the Panda's
randomised initial qpos (`robot_init_qpos_noise=0.02`), so "arm configuration
doesn't matter" becomes testable rather than assumed.

**Terminal and trajectory state: log it, because it cannot be regenerated.**
The rollout is stochastic — that episode happened once, and re-running the seed
produces a different one. Hence the terminal state dict and a stride-5 trace
(cubeA/cubeB/tcp xyz + the 4 flags) for every episode, stride-1 for the region
pass. This is what separates *locating* a failure region from *explaining* one:
`success_once` says an episode failed, the trace says whether cubeB was
displaced during the approach or during the placement, and those are two
different modes.

Not doing `RecordEpisode(save_trajectory=True)` during mining: `make_eval_envs`
attaches it to sub-env 0 only (`make_env.py:52`), so capturing all workers means
forking the thunk and giving up the bit-identical-to-training eval path that
makes these numbers comparable to T-I.

### Two conventions that differ from a naive reading

- **`relative_yaw_mod90`.** A cube has 4-fold yaw symmetry, so `+85°` and `−5°`
  are the *same* geometry. CLAUDE.md's pinned `(−π, π]` column is logged exactly
  as specified, but it is the wrong axis to regress against — it splits one
  physical configuration across two ends of the range. Analysis uses the mod-90
  column. Both are logged.
- **`face_gap` can go negative**, and that is meaningful rather than a bug: it
  means the two cubes' bounding squares overlap along the bearing between them,
  which the sampler's 58.6 mm centre-separation floor does not exclude for
  diagonally-presented cubes. Do not clamp it.
- **Wilson, not `sqrt(p(1−p)/n)`.** At n≈100 the interesting bins sit near
  p = 0, where the normal interval runs below zero and claims certainty it does
  not have — a 0/20 bin gets ±0.000 from the normal formula and [0, 0.161] from
  Wilson. Same data, honest bars.

### Pre-registration

The 80 mm threshold comes from the sampler analysis above, **written down before
any rollout ran**. The three refined thresholds — `face_gap < 25 mm`,
`dist_min < 520 mm`, `dist_max ≥ 760 mm` — are written down in
`notes/t2-failure-modes.md` and in `t2/run_modes.sh` before their passes ran,
along with the nominal-pass success rate each one has to reproduce on fresh
seeds. Every region pass draws its seeds from above the nominal range so the
passes are disjoint — re-measuring on the rollouts that found the region
measures noise.

The three refined filters also **partition on `dist_min` and `dist_max`**, so
they are mutually exclusive and each controls for the other modes' factor.
Without that they contaminate each other and none of the numbers means anything.

Say all of this in the report; a threshold fixed in advance is a materially
stronger claim than one chosen after seeing the scatter.

### A correction to this file

The success criterion above says cubeA must land "within ±5 mm in xy and z".
Source (`stack_cube.py:118-122`) uses `‖(0.02, 0.02)‖ + 0.005 ≈ 33.3 mm` in xy
and ±5 mm in z. The xy tolerance is nearly a full cube width, so "stacked but
visibly offset" scores as success. Worth knowing before calling a marginal
placement a failure.

**The gate on everything downstream** is a state-obs policy with success
meaningfully above 0 *and* meaningfully below 1. Above 0 is what PushT never
delivered. Below 1 is what PushCube would not have delivered — if StackCube
saturates, T-II has nothing to characterise and the task choice needs revisiting
immediately, not after the RGB run.

The RGB run's checkpoint is the T-I deliverable and the frozen base for T-IV.
Failure modes must be characterised on the *visual* policy, since that is what
T-III and T-IV improve.

**Decision point — decided: the report carries TWO targets, and they are chosen
to have different mechanisms.**

| | region | fails at | mechanism |
|---|---|---|---|
| **A** | `face_gap < 25 mm` | getting cubeA onto cubeB | approach fouls cubeB — displacement median 0.69 mm, 37.7% over 5 mm vs 16.6% reference; but `hold\|place` is normal at 0.809 |
| **B** | cubeB ≥ 760 mm from base, cubeA comfortable | the stack **staying** | places fine (0.870 given grasp, vs 0.904 reference) and `hold\|place` collapses to **0.657** vs 0.820; cubeB displacement is near-normal |

Two modes that fail at *different stages* by *different mechanisms* is a much
stronger T-II result than two slices of one curve, and it gives T-IV two
independent chances to show an effect. Budget for both properly — the original
warning still stands, that one mode with clean curves and a proper 3-seed eval
on both distributions beats two half-trained runs. If the budget will only carry
one, drop **B** and say so; **A** has the better-attested mechanism.

**Both are residual-fixable, and the earlier argument that reach was not has
been retracted.** The claim was that `a = a_base + clip(Δ, −α, α)` bounds Δ in
metres so it cannot extend the robot's reach. That is wrong twice:

- α bounds the **per-step** delta, applied over ~200 steps, so a persistent
  correction accumulates. The binding limit is not α's magnitude but the IK
  *saturating*, which only happens at the arm's actual kinematic edge.
- Mode B is not a reach failure at all. Conditioned on cubeA being comfortable,
  the far-cubeB grasp rate is **1.000** and the place rate is 0.870 — the robot
  gets there. It fails to make the stack settle, which is precision, which is
  what a bounded translation residual is for.

Where the original argument *does* hold is the both-cubes-far corner: grasp
drops to 0.879 above 720 mm and 0.815 above 740 mm (n=27), and no bounded
residual recovers a target the IK cannot reach. **Exclude that corner from
mode B's region** rather than letting it depress the result — hence "cubeA
comfortable" in the definition above, not just "cubeB far".

**The per-failure-mode number has a required shape: 100 rollouts × 3 seeds**,
same as T-I's. That means three **disjoint blocks of 100 region seeds**, each
run under its own policy seed — not one block of 100 evaluated three times.
Reusing one block and varying only the policy seed holds the initial states
fixed and measures DDPM sampling noise, which is a much smaller quantity than a
real error bar. `run_t2.sh do_region` does the former; it did the latter until
this was checked, and got the shape wrong (200 seeds × 2 repeats at one policy
seed).

The full 300-seed list per mode is also **the targeted-evaluation set** that
T-III biases toward and T-IV is scored on. `region_<tag>_seeds.csv` is that
artifact — fix it once and reuse it, or the before/after is not a comparison.

**Never cut:** the T-I baseline number, the quantitative failure
characterisation, and the nominal-distribution re-evaluation after finetuning.
Those three are the spine of the submission.
