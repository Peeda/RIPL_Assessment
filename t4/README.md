# T-IV — refining the frozen policy with a Policy Decorator residual

This directory answers one question: **can a small, bounded residual trained by
PPO against an LLM-written reward recover a specific, measured failure mode
without damaging everything else?**

The assignment's shape:

> Use the Policy Decorator method to implement a residual policy architecture
> (see Sec 4.1 and 4.2) … Finetune the residual policy using PPO with the
> LLM-generated rewards and episodic configuration from (T-III) … Report
> training curves (reward/loss/VRAM), wall-clock time, and any tricks you used
> to stabilize PPO training. Evaluate the full residual-augmented policy on the
> targeted failure-mode episodes and on the original full evaluation
> distribution. Report success rates over 100 rollouts and 3 seeds in both
> settings. The expected output is improvement on the targeted failure mode
> evaluations, and near-zero degradation on nominal performance.

Everything upstream is already fixed and must not move: the frozen base is
`checkpoints/stackcube_rgb_spatial_800demos.pt`, the two failure modes and
their 900 evaluation seeds are in `t2/results/`, and the reward and sampler are
in `t3/artifacts/{gap,farb}_gen2/`.

---

## 0. What was implemented, and what was not

Policy Decorator (Mu et al., ICLR 2025; code at `tongzhoumu/policy_decorator`)
is a **residual policy** learned online on top of a frozen imitation-learning
model, with **controlled exploration** to keep it stable. Taken from it:

| | from PD | where |
|---|---|---|
| residual on a frozen base, `a = a_base + a_res` | Sec 4.1 | `residual.py` |
| residual sees the base's **observation embedding**, not pixels | Sec 4.1, `actor_input='obs'` | `ResidualAgent.embed` |
| the residual emits a whole **action chunk** | their `single_action_space.low.repeat(act_horizon)` | `ResidualHead.out_dim` |
| **bounded** residual action, `alpha * tanh` | Sec 4.2, their `--res-scale` | `residual.bound` |
| progressive exploration over `H` steps | Sec 4.2, their `--prog-explore` | `residual.alpha_at` — **re-derived, see §2** |

Deliberately **not** taken:

- **SAC.** PD's backbone is SAC; the assignment specifies PPO. That single
  substitution is what forces §2, and it is the most interesting thing in this
  directory.
- **`--act-horizon 4`.** PD overrides the base policy's action horizon to
  re-plan more often. We cannot: it would make the base a *different* policy
  from the one T-I measured at 0.730 and T-II decomposed, and the entire
  deliverable is a paired before/after against those numbers. See §1.
- Their `--critic-input {res,sum,concat}` knob, which is a SAC artifact — PPO's
  critic is `V(s)` and never sees an action.

---

## 1. The base emits 8 actions per call; PPO wants one per step

`train_rgbd.Agent.get_action` predicts `pred_horizon=16`, returns
`act_horizon=8` (`noisy_action_seq[:, 1:9]`), and every rollout loop in this
repo executes all 8 open-loop before calling it again. PPO's per-step
assumption has to be reconciled with that, explicitly. **We lift the MDP to the
chunk.**

| | |
|---|---|
| state `s_k` | the frozen base's embedding at the k-th replan boundary, most recent frame only, `(256 + obs_state_dim,)` |
| action `a_k` | `res_horizon * 3` raw Gaussian; `Δ_k = α·tanh(a_k)` reshaped `(8, 3)` |
| reward `r_k` | **mean** of the 8 per-step `normalized_dense` rewards, in `[0, 1]` |
| transition | step the env 8 times with `base_chunk + Δ_k` |
| episode | `200 / 8 =` **exactly 25** chunked steps, no ragged remainder |
| `gamma` | **per chunk.** `0.9` here is `0.9**(1/8) ≈ 0.987` per env step |

**Why not per-step**, in order of how binding each reason is:

1. **A per-step MDP is not well defined without changing the base policy.** One
   action per observation means re-running the DDPM every step and keeping only
   its first action — that re-plans 8× more often and *is a different policy*.
   T-I's 0.730 and T-II's 0.513 / 0.623 would stop describing the "before" arm,
   and the paired comparison would collapse into two unrelated measurements.
2. **Cost.** ManiSkill's diffusion policy has no DDIM path —
   `num_diffusion_iters = 100`, always. Chunking amortises 100 U-Net forwards
   over 8 env steps (12.5/step); per-step would be 100/step, an 8× increase on
   the dominant cost of rollout collection, across six residuals.
3. **Credit assignment.** 25 decisions per episode instead of 200 shortens the
   span GAE has to carry advantage across by 8×.
4. It is what Policy Decorator does.

Two details that are easy to get wrong:

- **Mean, not sum, over the sub-steps.** With a constant chunk length the two
  differ by exactly a factor of `act_horizon` — a units choice, not a semi-MDP
  subtlety. The mean keeps `r_k ∈ [0,1]` and value targets O(1) with no reward
  scaling. `ignore_terminations=True` guarantees 200 is a chunk boundary so no
  partial chunk exists, but the loop still divides by the number of sub-steps
  **actually executed**: a silent partial chunk would be a units bug rather
  than a crash.
- **`num_steps` defaults to auto** = one episode. The right value moves with
  `--res-horizon`, and an iteration straddling an episode boundary makes every
  per-iteration curve harder to read for no gain.

### What chunking costs, and the knob that tests it

The residual is **open-loop for 8 env steps**. It cannot react to something
that goes wrong mid-chunk — cubeB beginning to topple at sub-step 3, say. Its
corrections are "aim the next 8 steps better", not "catch a slip".

For these two modes that is defensible: `gap` is about the *approach geometry*
of a descent and `farb` about *where and how high* the release happens, both
decided at a boundary. But it is an assumption, so `RES_HORIZON` exists to test
it rather than assert it. The base stays at 8 — frozen, byte-identical — while
the residual is re-queried every 4 or 2 env steps against a **fresh**
embedding, adding to the remainder of the base's plan. `ResidualAgent` holds
that remainder in `_pending`, so **the frozen base is still consulted exactly
once per 8 env steps**. Cost: one extra CNN forward per residual step, no extra
U-Net.

`get_action` returns `res_horizon` steps rather than 8, and every rollout loop
in the repo already sizes itself off `chunk.shape[1]` — so this needs no change
anywhere. What it does need is `reset_chunk()` at every env reset, which is why
that call now appears in `eval_modes.py`, `record_seeds.py` and `t3/check.py`.

---

## 2. Policy Decorator's exploration schedule cannot be ported to PPO

**This is the one place where following the paper literally would have been
wrong, so it is argued rather than asserted.**

PD Sec 4.2 defines the behaviour policy as

```
pi_behavior(s) = pi_base(s) + pi_res(s)     with probability eps
pi_behavior(s) = pi_base(s)                 with probability 1 - eps
```

with `eps` ramping linearly 0 → 1 over `H` steps. Their code is a Bernoulli
mask that zeroes the residual for the unselected envs.

That behaviour policy is a **mixture of a Dirac and a Gaussian**. Its density
is undefined at the atom, so there is no `log pi_behavior(a|s)` — and PPO's
entire update is an importance ratio `pi_new / pi_old` built from exactly that
quantity. SAC never forms one (it is off-policy and reads actions out of a
replay buffer), which is why the schedule is free there and unusable here.

Silently keeping the mask would not crash. It would put a large population of
`Δ = 0` actions into the batch, scored under the Gaussian's density as if the
policy had chosen them — a systematically wrong ratio, worst exactly early in
training when `eps` is small. That is the kind of bug this repo exists to
refuse.

**The PPO-safe analogue ramps the bound instead:**

```python
alpha_t = alpha * min(env_step / H, 1)      # residual.alpha_at
```

It buys the same thing — the residual starts unable to deviate and is let out
gradually — and the log-probs stay exactly correct, because **α does not enter
the log-prob at all**. The raw Gaussian sample *is* the action; `α·tanh` is
part of the transition. What would break if α moved *within* an iteration is
not the ratio but the MDP: returns and value targets would describe two
different environments. So α is computed once per iteration and held across the
whole collect-and-update. Between iterations it is fine — PPO tolerates slow
non-stationarity, which is the whole idea of the schedule.

Together with near-zero actor initialisation (§4), the residual at iteration 0
is *exactly* the base policy, and the bound opens from 0 mm to 5 mm over the
first 1 M env steps.

---

## 3. The gripper dimension is excluded from the head, not masked after it

`pd_ee_delta_pos` is 4-dimensional: 3 translation + 1 gripper. The residual
head emits `res_horizon * 3`. There is **no fourth coordinate anywhere** — no
output row in the final `Linear`, no `log_std` entry, no sampled dimension. The
scatter is `a[..., :3] += delta`.

The alternative — emit 4 per step and zero the last after sampling — is wrong
for a reason bigger than the ~2 k wasted parameters. Those inert coordinates
would still enter `log_prob.sum(-1)`, `entropy()` and `approx_kl`, so PPO would
integrate noise over dimensions that provably cannot move the simulator:
entropy inflated, extra variance in the importance ratio, and `target_kl`
early-stopping triggered by movement in dimensions nothing depends on.

**Why exclude it at all** (CLAUDE.md's fixed decision): a bounded perturbation
on the gripper channel at the wrong moment simply drops the cube — degrading
the base through a channel unrelated to the spatial correction the residual
exists to learn. It would also destroy α as a bound in metres, since dim 3 is
not a displacement.

### What the residual can and cannot change

**Directly, it can never change the gripper command.** `a[..., 3]` is whatever
the frozen base emitted, at every timestep.

**Indirectly it changes gripper *timing*, because the base is closed-loop.**
The base re-plans every 8 steps from the current observation; move the end
effector and its next chunk — gripper channel included — is conditioned on a
different state. The residual controls *where the arm goes*; grasp and release
timing follow from that through the base policy. It cannot override a grasp
decision *within* a chunk.

Two honest caveats:

- Both modes' failures are spatial, so this is not obviously binding. `gap`
  fails at **placement** (the descent onto cubeA fouls cubeB) — purely
  translational. `farb` fails at **settling**, and the levers there are also
  translational: centre better in xy, release from a lower height, retreat
  without brushing the stack.
- **But both gen2 rewards pay explicitly for gripper opening** — `open_frac` in
  `farb`'s `r_held`, `ungrasp` in `gap`'s stage C. Part of the reward's gradient
  therefore lies on a channel the residual does not control. Not fatal, since
  the reward still *ranks* states correctly (which is what T-III's alignment
  measurement checks), but it is a real limitation and it should be reported
  rather than discovered.
- `res_dim` is a checkpoint field, so a 4-dim ablation is available if `farb`
  shows no improvement and the question is whether the gripper channel is what
  binds. Run it as an ablation, never as the headline number.

---

## 4. α is a bound in metres

`pd_ee_delta_pos` maps a normalised ±1 to ±0.1 m (`panda.py:105-106`), so

```
alpha = 0.05   ->   5 mm per env step
```

PD's default `--res-scale` is 0.05 and their StackCube entry is 0.03; the sweep
is `{0.03, 0.05, 0.10}` on `gap` seed 1 only, chosen on **training** success
and `delta_norm_mm` — never on the evaluation set, which would make the
deliverable a training number.

The retracted argument is worth keeping: α does **not** bound how far the arm
can ultimately move. It bounds the *per-step* delta, applied over ~200 steps,
so a persistent correction accumulates. The binding limit is the IK saturating,
which only happens at the arm's actual kinematic edge — which is exactly why
`geometry.MODES["farb"]` carries `dist_A < 0.72`, excluding the both-cubes-far
corner where no bounded residual could recover a target the IK cannot reach.

`charts/delta_norm_mm` is plotted against `charts/alpha_mm` on the same axes in
`t4_training.png`. If the measured `|Δ|` sits on the bound, α is what binds and
the sweep matters; if it sits well below, α is not the constraint and the knob
to turn is elsewhere.

---

## 5. The seam into `t2/`, and why the residual is a separate file

Most of `t2/` is policy-agnostic and transfers untouched: `select_seeds` is
deterministic given `seeds.csv`, the four per-episode reset assertions,
`geometry.COLUMNS`, `verify.py`, `report.py`. Exactly one piece did not, and
the fix is small.

**`build_agent` now always returns a `ResidualAgent`, even with no residual.**
CLAUDE.md: *"Both arms must go through that one path, or the before/after
compares two code paths as well as two policies."* With `head=None` the wrapper
adds no tensor operation and returns the base's own chunk, so the before arm is
bit-identical to what every committed T-II number was measured on —
`t2/verify.py` still exits 0 on `t2/results` and `test_geometry.py` still
reports 303.

**The residual is a separate small file behind a `RESIDUAL` env var, never
merged into the base state dict.** This is what lets `inspect_ckpt`,
`load_weights` and `verify.py` check 5 ("exactly one `ckpt_sha256` per pass")
keep working untouched: `CKPT` stays the frozen base, and check 5 keeps meaning
what it meant. A merged checkpoint would have needed all three relaxed.

**`RESIDUAL` may contain `{block}`**, because T-IV pairs residual seed *b* with
evaluation block *b*. `eval_modes.py` attaches per block and records
`residual`, `residual_sha256`, `residual_{mode,seed,alpha,res_horizon}` in that
block's manifest; with no residual the manifest keeps exactly the keys it had.

`get_action` calls `base.get_action(dict(obs))` **first** — which permutes rgb
and runs the DDPM — then `base.encode_obs` on its own permuted copy.
`encode_obs` consumes no RNG (PlainConv is conv/relu/pool/fc, and `self.aug` is
dead code in this baseline), so the base action is bit-identical whether or not
a residual is attached, and the caller's observation dict is never mutated.

### Two designs for "3 seeds", and which one we ran

T-I and T-II read "100 rollouts × 3 seeds" as three disjoint blocks of 100
region seeds under three policy seeds, and `verify.py` check 6 enforces that
shape. For T-IV that leaves the usually-dominant source of variance —
**training** — unmeasured: three numbers differing only in DDPM sampling noise
on a fixed set of initial states.

The **paired** design fixes that: train three residuals, evaluate seed *b* on
block *b*, so each of the three numbers is an independent replicate of the whole
pipeline. Evaluation cost is identical either way — only training triples.

That was affordable on GPU and is not on CPU, so **we ran the single-residual
design**: one residual per mode, evaluated over all three blocks. Both are
supported by the same code (`RESIDUAL` with or without `{block}`), and
`verify_t4.py` distinguishes them, refuses the incoherent middle case (two
distinct residuals across three blocks, which is a mis-set `$RESIDUAL` far more
often than a choice), and prints the caveat when the single design is detected.

---

## 6. Training runs on `physx_cpu`, because the gate said so

The plan was to train on `physx_cuda` and score on `physx_cpu`, because CPU
looked far too slow to carry a multi-million-step PPO run.
`t4/backend_check.py` was built to license that split.

*(That "too slow" figure was ~28 env-step/s, and it was wrong — it read
`wall_seconds` in a block manifest as the time for that block, when it is
CUMULATIVE across the pass. The T-II evaluation actually ran at **112
env-step/s** on 10 processes: nine blocks of 100 episodes in 1611.8 s, ~179 s
each. The gate's verdict does not depend on it — CPU training measures ~138
env-step/s at 64 processes, still ~40× the width GPU would have given — but the
number is corrected here so it is not planned on again.)*

**It refused.** Replaying the 300 committed nominal initial states on
`physx_cuda` with the frozen base:

| | physx_cpu | physx_cuda | diff |
|---|--:|--:|--:|
| `success_once` | 0.730 | **0.557** | **−0.173** |
| `ever_grasped` | 0.973 | 0.987 | +0.013 |
| `ever_placed` | 0.860 | 0.783 | −0.077 |
| `success \| placed` | 0.849 | **0.711** | **−0.138** |

paired agreement **0.667** against a ~0.74 same-backend floor, 76 cpu-only
against 24 gpu-only, McNemar χ²=26.0, p<0.001.

**Perception is fine** — the grasp rate *holds*, in fact rises slightly. A
rendering difference would have shown there, because the base policy is visual
and grasping is the stage that depends on localising the cube. The loss is
contact physics, concentrated after placement. It is not a config difference
either: `SceneConfig`'s solver iterations are shared, `sapien_env.py` does not
branch on backend, and StackCube does not override `_default_sim_config`.

**What settles it** is *where* the loss lands. T-II's `farb` mode is defined as
"places at near-baseline rates, then the stack does not settle" — `hold|place`
0.657 against nominal 0.820. GPU physics manufactures that same failure
globally. A `farb` residual trained there would be learning to fix a backend
artifact, and neither a positive nor a null result could be attributed to the
method.

So training is `physx_cpu`, and the budget shrank to fit: **one residual per
mode instead of three**, ~1 M env steps instead of 4 M. The three evaluation
blocks then vary only the initial states and the DDPM noise, so **the reported
SD does not carry training variance** — `verify_t4.py` detects the
single-residual design and prints that caveat, and the report must repeat it.

`physx_cpu` raises `RuntimeError` for `num_envs > 1`, so `make_envs.py`'s CPU
branch delegates to the diffusion-policy baseline's own `make_eval_envs` — the
literal function `t2/harness.py` builds the scoring env with, so the training
and scoring pipelines cannot drift. `--num-envs` is then a **process count**.

Two backends means two info conventions, and `train_ppo.episode_end` normalises
them: `ManiSkillVectorEnv` gives one batched dict, `gym.vector.AsyncVectorEnv`
an object array of per-env dicts. Reading one as the other does not raise — it
silently averages the wrong thing — so both are covered by
`test_train_loop.py`, which runs the whole loop once per backend.

### What the check does, and how to read it

**A seed does not address an episode on `physx_cuda`**, and the assertion
that should catch that **passes and lies**: `_initialize_episode` draws cube
poses under `with torch.device(self.device)` so the CUDA generator gives
different states for the same seed, `reset()` seeds the whole batch from
`_episode_seed[0]`, and `_episode_seed` is nonetheless recorded per env. The
seed index and every T-II seed list are `physx_cpu` artifacts.

So: train on GPU, score on CPU. `t4/backend_check.py` (recovered from
`c1c010b^`) is what licenses the split. It sidesteps seeds entirely —
`capture_states.py` reads the exact `physx_cpu` initial states of the 300
committed nominal episodes (no policy, minutes), and `backend_check.py` replays
them on `physx_cuda` through `options={"reset_to_env_states": ...}`, proving
each injection took by reading cubeA back against the CSV.

**The verdict is not "agreement == 1.0".** DDPM sampling is stochastic, so
identical initial states disagree even CPU-vs-CPU; that floor was measured at
~0.74. The questions are whether GPU-vs-CPU lands *at* the floor, whether the
disagreement is symmetric (McNemar) or directional, and — separately, and this
is the one that matters — whether the **conditional** table over `face_gap` and
`dist_max` moves. The marginal agreeing while the failure region shifts is the
outcome that would invalidate everything downstream, and only that table
catches it.

**Run it on any new pod before trusting a backend.** `bash t4/run.sh backend`.
It is also, on its own, a reportable result: a measured refusal to train across
a 17-point dynamics gap is a stronger methods paragraph than a silent
GPU run would have been.

`simstate.py` holds `flatten_state_dict` / `state_dict_from_flat` once, shared
by both scripts, because a flattener and an unflattener that disagree fail
silently by construction. The trap they exist for: flattening walks **sorted**
keys for stability across ManiSkill versions, while `set_state` reconstructs in
**insertion** order, so a positional round trip reads cubeB's pose out of
cubeA's slot and the episode simply starts somewhere else.

---

## 7. The training distribution, and the mixing knob

`T3_SAMPLER=1` replaces StackCube's own cube placement with T-III's. Measured
here (4096 draws, `torch.manual_seed(0)`, base rate from the committed
25,000-row index — see `t3/artifacts/README.md`):

| sampler | hit rate | nominal base | enrichment |
|---|--:|--:|--:|
| `gap_gen2` | 0.897 | 0.0468 | 19.2× |
| `farb_gen2` | 0.996 | 0.0337 | 29.5× |

So ~90–99% of training episodes are in the region, against a 3–5% base rate.
That is the point — almost every gradient step is about the thing being fixed.

The risk it creates is the assignment's own second criterion. The residual
fires on **every** state at evaluation, including the ~95% of nominal episodes
that look nothing like training. PD's defence is the bounded residual alone: α
caps the correction at 5 mm/step, so the damage is bounded but not zero.

`T4_NOMINAL_FRAC` is the second line, and it **defaults to 0.0** because "the
episodic configuration from T-III" is what the assignment specifies. At 0 the
blending branch in `env_t4.py` is not even entered and the env is behaviourally
`StackCube-T3-v1`. Raise it to 0.5 only if the nominal arm actually degrades,
and then report **both** runs — the comparison is a result either way: "the
bound was sufficient" or "it was not".

The blend needs to read cube poses back mid-`_initialize_episode`, which is
version-sensitive, so `_rows` fails loudly rather than risk one env's cubes
landing in another's initial state.

**Training uses no seeds from the evaluation blocks — it uses no seeds at
all.** With the sampler on, `reset(seed=s)` does not produce the state
`seeds.csv` records for `s`; the training distribution is continuous and shares
no episode with the 900 evaluation seeds. The overlap is distributional, which
is exactly what T-III was for.

---

## 8. The stabilisation tricks, and the metric that shows each working

The assignment asks for these by name.

| trick | why | what shows it |
|---|---|---|
| **bounded residual** `α·tanh` | a hard cap on how far from the base the policy can get, per step | `charts/delta_norm_mm` against `charts/alpha_mm` |
| **α ramp** (§2) | PD's progressive exploration, in the only form PPO admits | `charts/alpha_mm` |
| **near-zero actor init** (`std=0.01`, `log_std=−1`) | iteration 0 *is* the base policy; PPO never climbs back from a random start | `train/success_once` starts at the base rate, not at 0 |
| **frozen encoder as the feature extractor** | ~400 k trainable parameters and no gradient into the vision stack the base depends on | `head_params` in the manifest |
| **reward pre-normalised to [0,1]/step** | `REWARD_MAX` division in `env_t3`; no reward scaling to tune | `charts/reward_mean` bounded |
| **advantage normalisation, grad clip 0.5, KL early-stop 0.1** | upstream `ppo.py` defaults, kept | `losses/approx_kl` against its line, `losses/clipfrac` |
| **one full episode per iteration** | GAE never crosses an episode boundary | `--num-steps` auto |

If PPO is unstable, in this order: raise `--num-envs`; drop `--learning-rate`
to 1e-4; lower α to 0.03; then an asymmetric critic (privileged cube poses into
`V(s)` only — legitimate, since the critic is discarded at evaluation).

---

## 9. Running it

```bash
source /workspace/ripl/env.sh
bash setup/apply_patches.sh            # setup_runpod.sh re-clones and wipes it

bash t4/run.sh test                    # laptop; needs torch + tyro, see the file header
MODE=gap bash t4/run.sh backend        # the gate; re-run it on any new pod
MODE=gap bash t4/run.sh smoke          # one tiny PPO iteration end to end
NUM_ENVS=$(nproc) MODE=gap bash t4/run.sh train    # in tmux, tee to a file
MODE=gap bash t4/run.sh eval           # the paired T-II evaluation, physx_cpu
MODE=gap bash t4/run.sh verify         # t2/verify.py, then verify_t4.py
bash t4/run.sh report
```

**Measure the training rate before sizing the budget.** `smoke` prints
`env-step/s`; multiply out rather than trusting the ~28/s figure from a
10-process evaluation, and check it against a stopwatch — CLAUDE.md records a
printed throughput being wrong by 10× on PushT. `--num-envs` is a process count
on `physx_cpu`, so `nproc` is the ceiling.

Then the same with `MODE=farb`. `run.sh` handles the two operational traps
itself: the after pass writes to its own `$T2_OUT` (pointing it at
`t2/results` would skip every finished block and silently reprint the base
numbers — `FORCE=1` is the wrong fix), and `seeds.csv` is **copied** rather
than rebuilt (selection is deterministic *given the index*, so a differently
sized index selects different seeds and the comparison stops being paired).

| stage | needs | cost |
|---|---|---|
| `test` | torch, tyro | ~8 s |
| `capture` | ManiSkill, physx_cpu | ~3 min |
| `backend` | ManiSkill, physx_cuda | ~15 min |
| `smoke` | ManiSkill | ~2 min |
| `train` | physx_cpu, GPU for the policy | measure it — see below |
| `eval` | ManiSkill, physx_cpu | ~1.8 h per mode (9 blocks) |
| `verify`, `report` | stdlib / numpy+matplotlib | seconds |

---

## 10. What is NOT checked

In `t3/README.md` §4's spirit: a gap you name is a finding, a gap you imply is
covered is a mistake.

1. **That the residual helps for the reason the reward says.** Every number
   here is `success_once`. The stage flags (`ever_grasped`, `ever_placed`,
   `ever_static`) are logged and the mechanism split can be read off them, but
   nothing asserts that `gap`'s recovery came from clearing cubeB during the
   descent rather than from some unrelated change. The `cubeB_displacement`
   column is the closest available evidence.
2. **Reward hacking against the *generated* reward.** T-III's alignment check
   established that the reward ranks *the base policy's* episodes correctly. It
   says nothing about whether a trained residual finds a way to accumulate that
   reward without stacking. The only thing standing between us and that is
   scoring on `success_once` under the environment's own sparse criterion,
   which is exactly what `t2/eval_modes.py` measures — a residual that farms
   the shaped reward will show it as a flat or falling success rate against a
   rising `charts/reward_mean`. **Read those two panels together.**
3. **Whether the gripper channel is what binds.** §3. Testable via `res_dim`,
   not tested by default.
4. **Cross-mode transfer as anything but a bonus.** The `eval` stage runs all
   three arms for each residual, so "the `gap` residual does not help `farb`"
   is measured — but n=300 per cell means only a large effect is visible.
5. **Training-time distribution shift within a mode.** The sampler's hit rate
   is 0.897 for `gap`, so ~10% of training episodes are outside the region the
   residual is scored on. That is left as generated, deliberately.

`train_ppo.py` is the one file that cannot be tested for real off-pod, so
`test_train_loop.py` stubs gymnasium and ManiSkill, stands in a fake base with
the same surface `train_rgbd.Agent` presents, and runs `main()` end to end. It
proves nothing about learning — the fake env has no dynamics — but it proves
every tensor lines up, GAE and the update run, the alpha ramp reaches the
bound, the checkpoint round-trips through `t2/harness`'s loader, and the CSV
carries every column `report.py` reads. Mutation-checked: summing instead of
averaging the sub-step rewards, counting residual steps instead of env steps,
and dropping the ramp each fail a named assertion.

Bernoulli SE at n=100 is ~5%, and the three-block SD is the honest error bar.
**"Near-zero degradation" is a claim that 3 × 100 supports only weakly.** Say
so in the report, as T-II did.

---

## 11. The files

| | needs | |
|---|---|---|
| `residual.py` | **torch only** | the head, the α bound, the α ramp, `ResidualAgent`, the checkpoint format |
| `env_t4.py` | ManiSkill | `StackCube-T4-v1`, `T4_NOMINAL_FRAC` |
| `make_envs.py` | ManiSkill | the GPU training env, wrapper order copied from the DP baseline |
| `train_ppo.py` | ManiSkill + torch | PPO, adapted from `examples/baselines/ppo/ppo.py` |
| `capture_states.py` | ManiSkill | physx_cpu initial states for the backend check |
| `backend_check.py` | ManiSkill | recovered from `c1c010b^`; the GPU/CPU licence |
| `simstate.py` | torch + numpy | flatten/unflatten a sim state, once |
| `verify_t4.py` | **stdlib** | the pairing and residual-provenance checks; exits non-zero |
| `report.py` | numpy + matplotlib | the two figures, the tables |
| `test_t4.py` | torch | 193 assertions, no simulator |
| `test_train_loop.py` | torch + tyro | runs `train_ppo.main()` against a STUBBED ManiSkill |
| `run.sh` | bash | the one driver |

The same layering as `t2/` and `t3/`: what *defines* the method is testable
with the cheapest possible feedback, what needs a simulator says so and fails
loudly off-pod, and what reads the measurements back needs neither.

The debugging narrative goes in `notes/t4-residual-ppo.md`.
