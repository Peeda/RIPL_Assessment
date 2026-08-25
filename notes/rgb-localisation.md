# The RGB policy could see, but could not aim

Report material, and the second debugging narrative in this repo after
`notes/pusht-detour.md`. That one is the record of a task that was abandoned.
This one is the record of a task that worked, via one wrong diagnosis, one
expensive-but-correct lever that ran out, and one architectural fix.

The short version: a state-observation diffusion policy on `StackCube-v1`
reaches the cube to **4 mm**. The identical recipe on RGB observations reached
it to **86 mm** — against a **40 mm** cube — and succeeded 4% of the time. The
cause was not the amount of data, not the training budget, and not blindness.
ManiSkill's baseline visual encoder global-max-pools its feature map before the
FC, which discards every spatial coordinate, and on StackCube spatial
coordinates are the *only* thing the encoder has to supply.

---

## Where it started

Both numbers below come from `t2/rollout_log.py` against
`best_eval_success_once.pt`, same replayed demonstrations, same control mode
(`pd_ee_delta_pos`), same 100 demos. The runs differ only in `--obs-mode`.

| | state | rgb |
|---|---|---|
| success | **0.680** | **0.040** |
| mean reach error to cubeA | **4 mm** | **86 mm** |
| corr(cubeA_x, tcp_x) | +0.999 | +0.689 |
| corr(cubeA_y, tcp_y) | +1.000 | +0.819 |
| reach-spread / cube-spread | 1.00× | 0.79× (x), 0.84× (y) |

"Reach" is TCP xy at deepest descent — where the policy decided to grasp.

> **Caveat that travels with these two success figures.** They are labelled
> `success_once` in the original notes and are really `success_at_end`. The
> pre-fix `rollout_log.py` read the top-level `info['success']` at the final
> step, but `CPUGymWrapper(record_metrics=True)` nests the real `success_once`
> under `info['episode']`, and `ignore_terminations=True` runs every episode to
> the full horizon — so the top-level key is the end-of-episode value. Since
> `success_once >= success_at_end` always, the true values are *at least* 0.680
> and 0.040. The harness logs both columns now. **Never compare a pre-fix
> rollout number against wandb's `eval/success_once`.**

Two things about the state row are worth stating before moving on, because they
frame everything after it. First, 0.680 is a *good* T-I baseline — meaningfully
above 0, meaningfully below 1, which is exactly what T-II needs. Second, a
policy that localises to 4 mm still failed 32% of the time, so those failures
happen strictly downstream of reaching: grasp, place, release, settle. The
visual gap and the manipulation gap are separate problems, and only the first
one is what this note is about.

---

## The wrong diagnosis: "it ignores the image"

The eval videos show an arm that descends confidently and closes on nothing.
The obvious reading is that the visual encoder is being ignored and the policy
is replaying an average trajectory — behaviour cloning that learned the *shape*
of the motion but not where to point it.

A probe was written for exactly this (`probe_visual_dependence.py`, now deleted
— see below): hold proprioception fixed, swap only the image between
observations from different initial states, and see whether the predicted action
moves. **The reading was wrong.** Correlation between cube position and reach
target is 0.69–0.82. That is real visual tracking. The policy is not blind.

The instructive part is that correlation was *also* misleading in the other
direction, and for a while it was quoted on its own as if 0.82 were reassuring.
Pearson's r is invariant to scale and offset, so r = 0.82 sits perfectly happily
beside an 86 mm error. Decomposing the error settles it:

- shrinkage toward the training-set mean is **mild** — the reach spread is
  0.79–0.84× the cube spread, not 0.1×;
- the error standard deviation is **σ ≈ 109 mm combined**.

So the policy aims in roughly the right direction and lands most of a cube-width
away. It is coarse localisation, not blindness. **Always quote the error in mm
alongside r.** Reach error turned out to be the better instrument throughout: it
is continuous, it has a physical scale against a 40 mm cube, and it separates a
perception failure from a manipulation failure — none of which a success rate or
a correlation does.

`probe_visual_dependence.py` has been deleted from the repo. It answered a
binary question with a coarse three-way verdict, `rollout_log.py`'s reach error
answers the same question better and quantitatively, and keeping the tool that
produced the wrong reading next to the tool that produced the right one only
invites re-running it. It is in git history if it is ever wanted.

---

## The data lever, and where it stopped

The next hypothesis was memorisation. A visual policy has to learn perception
from roughly 100 demos × ~110 frames ≈ **11k frames**, which is not obviously
enough for anything. The replay produces ~990 trajectories and the recipe uses
100 of them; the other ~890 were already sitting on disk. `--num-demos` is a
free lever — it does not touch the LR schedule and costs only dataloading and
memory.

`baselines.sh`'s StackCube rgb line with `--num-demos 800`, everything else
identical:

| demos | encoder | success_once |
|---|---|---|
| 100 | pooled (upstream) | 0.040 |
| 800 | pooled (upstream) | **~0.43** (final eval 0.43, best 0.49) |

**The memorisation hypothesis was right**, and 0.04 → 0.43 is a real result. But
the lever is now spent, and it is worth being precise about why, because "try
more data / more steps" is the reflex that would otherwise eat the next 100k
iterations too:

- **The run converged.** ~21 eval points at `eval_freq=5000` over the full 100k
  schedule. Flat from step 60k — the last 40k iterations oscillate 0.42–0.49
  with no trend, and at `--num_eval_episodes 100` the Bernoulli SE is ≈5%, so
  that entire oscillation is one standard error. At 800 demos (~88k transitions,
  batch 256) 100k iters is ~290 epochs.
- **More demos are not available.** 800 is most of what the replay produces.
- **The run could not have been extended anyway.** `total_iters` is an LR
  hyperparameter here, not a budget: `train.py` uses a diffusers cosine schedule
  with 500 warmup steps and `num_training_steps=total_iters`, so `--lr 1e-4` is
  the *peak* and the LR anneals to 0 at `total_iters`. Raising it resumes under
  a different LR trajectory; truncating it leaves the weights at 80% of peak LR
  with none of the low-LR annealing. A long run cannot be early-stopped into a
  short one, and a short one cannot be extended into a long one.

That leaves a gap of 0.43 vs the state policy's 0.680 with no data lever and no
budget lever remaining. The gap had to be representational.

---

## The real diagnosis: the encoder throws position away

`train_rgbd.py:272-274` hardwires the visual encoder:

```python
self.visual_encoder = PlainConv(
    in_channels=total_visual_channels, out_dim=256, pool_feature_map=True
)
```

and that flag selects `nn.AdaptiveMaxPool2d((1,1))` (`plain_conv.py:56-58`). A
128×128 image goes through four max-pools down to an **8×8×128** feature map,
and the whole 8×8 grid is then collapsed to **one number per channel** before
the FC to 256. There are no spatial coordinates left in the representation.
Position survives only as "which of 128 channels fired" — a coarse code with no
spatial support.

That is *exactly* the measured signature: real tracking (r = 0.69–0.82) with a
large scatter (σ ≈ 109 mm) rather than a bias, against a 40 mm target.

It binds on `StackCube-v1` specifically, and this is the part that makes it a
task property rather than a general complaint about the baseline. Under
`--obs-mode rgb` the state vector carries no cube pose at all
(`stack_cube.py:133-143`):

```python
obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
if "state" in self.obs_mode:          # <- not taken under rgb
    obs.update(cubeA_pose=..., cubeB_pose=..., tcp_to_cubeA_pos=..., ...)
```

So the RGB policy sees `tcp_pose` + `qpos/qvel`, and **100% of cube localisation
must flow through the pooled encoder**. The state policy reaches to 4 mm because
it is simply handed the pose. The gap between 0.68 and 0.43 was never "vision is
harder" in the abstract — the single quantity vision has to supply is the one
quantity the encoder discards.

---

## The fix, and why it is shaped the way it is

`patches/0001-dp-rgb-spatial-feature-map.patch` makes `pool_feature_map` an
`Args` field defaulting to `False`, rather than flipping the hardcoded constant.
Three reasons, all of which turned out to matter:

- both arms stay runnable, so the A/B is a flag rather than a git checkout;
- the value lands in wandb's config through `vars(args)` (`train_rgbd.py:453`),
  so a run's encoder variant is recorded with the run rather than remembered;
- `run_pipeline.sh` puts the variant in the run directory name, so the two
  checkpoints cannot be confused on disk.

With pooling off, the FC input goes from **128** to **8192** (= 128 × 8 × 8).

Two things not to "fix" while in there:

- **`128*4*4*4` in the non-pooled branch is already correct.** It equals
  `128*8*8 = 8192` for a 128×128 input. Only the inline `[4,4]` shape comments
  are stale.
- **Checkpoints are not interchangeable across this flag.**
  `visual_encoder.fc.0.weight` is `(256, 128)` pooled and `(256, 8192)` spatial.
  Rather than adding a manual flag that can be set wrong, `t2/t2_common.py`'s
  `inspect_ckpt` infers the variant from that weight's shape and configures
  `Args` to match — the same way it infers obs mode from whether any
  `visual_encoder.*` keys exist at all.

### Deviating from the published recipe cost nothing here

The standing objection to touching a documented hyperparameter is comparability
to published numbers. It does not apply.
`docs/source/user_guide/learning_from_demos/baselines.md` lists Diffusion Policy
for StackCube as **Results: WIP** — as it does for BC, ACT, RFCL and RLPD. So
`baselines.sh` is a set of *tuned commands*, not a set of *verified results*, and
there is no maintainer figure to stay comparable to. The recipe buys a sane
starting point; it does not buy a target.

The consequence runs both ways, and the second half is the uncomfortable one: a
low number could not have been called a bug by comparison either. It could only
be judged against what the assignment needs, which is a base policy meaningfully
above 0 and meaningfully below 1.

---

## Result

| demos | encoder | success_once | reach error |
|---|---|---|---|
| 100 | pooled (upstream) | 0.040 | 86 mm |
| 800 | pooled (upstream) | ~0.43 (best 0.49) | — |
| 800 | **spatial** (`patches/0001`) | *provisional, see below* | — |

**The spatial row is not yet a result.** The only figure in hand is ≈0.87 from a
*scaled-down* shakedown pass of the T-II harness at n ≈ 76 episodes — a wide
interval, on a seed block that overlaps nothing the production pass will use, run
to prove the harness works rather than to measure the policy. It is quoted here
only so the direction is on record; **no number from that pass belongs in the
report.**

The real row comes from `bash t2/run_t2.sh t1`: 3 × 100 episodes on three
disjoint seed blocks with three policy seeds, which is the assignment's "100
rollouts × 3 seeds" and is also the only source of T-I error bars, since at high
`--num_eval_envs` wandb logs a single mean per eval and per-eval variance is not
recoverable from the training logs.

One more selection effect to expect when that lands: `best_eval_success_once.pt`
is the max over ~21 evals at n=100 with SE ≈ 5%, so some of "best" is luck. A
fresh-seed re-evaluation should come in a little *below* the eval number that
selected the checkpoint. That is the bias unwinding, not a regression, and it is
precisely why the reported number comes from an independent harness.

---

## What this hands to T-II — and one open worry

The spatial checkpoint is the T-I deliverable and the frozen base for T-IV, so
the failure modes T-III and T-IV target must be characterised on *it*, not on
the state policy.

**The worry is that the fix worked too well.** At ~0.43 there were failures
everywhere and T-II's problem was choosing among them. If the visual policy
really sits near 0.87, then a 100-episode nominal pass yields ~13 failures, and
a per-region success rate estimated from a handful of episodes is not a
characterisation. Two things already in the design absorb this, which is why it
is a worry and not a problem:

- the failure region is targeted by **resampling fresh seeds** from it
  (`separation < 80 mm`, pre-registered from the sampler analysis before any
  rollout ran), so in-region episode count is set by budget, not by how often the
  nominal distribution happens to produce one;
- `success_once` vs `success_at_end` is logged separately, and their gap is its
  own candidate failure mode — stacked, then toppled — which does not depend on
  the overall rate being low.

Resolving it is T-II's job, not this note's.

---

## What I'd do differently

- **Measure reach error before spending a 100k-iteration run on a data lever.**
  The 800-demo run was the right call and produced a real result, but the
  encoder finding was available from source-reading plus a 50-episode probe, and
  it would have reframed the data run from "the fix" to "one of two levers".
- **Read the model before believing "more data".** The pooled max-pool is four
  lines of `plain_conv.py`. Every hypothesis entertained before reading it —
  memorisation, blindness, insufficient iterations — was about *quantities*,
  because nothing had checked whether the architecture could represent the thing
  being asked of it.
- **Quote a physical error next to every correlation.** r = 0.82 read as "vision
  works, keep training". 86 mm against a 40 mm cube read as "the policy cannot
  hit the cube", which is the same fact and the actionable version of it.
- **`self.aug` is dead code and nobody noticed for a long time.**
  `train_rgbd.py:301` guards it with `hasattr(self, "aug")` and nothing ever
  assigns it, so there is **no image augmentation anywhere in this baseline**.
  That is the obvious untried lever if the spatial encoder had not been enough,
  and it was invisible until someone read the file for a different reason.
