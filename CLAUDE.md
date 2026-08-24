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
not overhead. `notes/pusht-detour.md` is the first of them.

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
| Task | `StackCube-v1`, `motionplanning` demos | One raw `trajectory.h5`, a published DP recipe, and — unlike `PushCube-v1` — not saturated, so T-II has failures to characterise. |
| Control mode | `pd_ee_delta_pos` (**4-dim: 3 translation + gripper**) | Matches the published recipe. T-IV's α is only a coherent physical bound when what it bounds is a displacement in metres — see the residual row. |
| Replay flag | **`--use-first-env-state`** | Motionplanning demos are recorded under `pd_joint_pos`, so replay is a control-mode *conversion* and must simulate forward from the initial state. **This reverses the PushT rule**; see below. |
| Backend | `physx_cpu` everywhere | Matches the published recipe. Replay and eval backends must still match each other — the backend is in the output filename for that reason. |
| T-IV residual | 3 translation dims only; gripper passes through | See below. |
| Failure filtering | never `--allow-failure` | Training IL on failed demonstrations is worse than training on fewer. |
| Demos / steps / iters | 100 / README / README | Copy the baselines README verbatim. Do not invent hyperparameters; documented ones are why this task was chosen. |

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

## Open questions — resolve on the pod before the first real run

These are unverified. Read them off the source rather than trusting this file.

1. **Does `download_demo StackCube-v1` serve motionplanning demos?** If not,
   generate them: `python -m mani_skill.examples.motionplanning.panda.run -e
   StackCube-v1 --num-traj 1000`. `run_pipeline.sh` tries the download and falls
   back, printing which ran.
2. **The published hyperparameters.** `run_pipeline.sh`'s `MAX_EP_STEPS` and
   `TOTAL_ITERS` are placeholders. Copy the StackCube line from
   `$MANISKILL_REPO/examples/baselines/diffusion_policy/README.md` and patch both
   the script and this table.
3. **StackCube's initial-state sampler.** Read `_initialize_episode` in the task
   file and write down the actual support. It is believed to reject
   configurations where the cubes spawn closer than some minimum XY distance — if
   so the default distribution is truncated *exactly where the failures are*.
   That is not a problem; it is T-III's job description, and it means T-III's
   sampler has real work to do rather than merely reweighting.
4. **Eval cost at 100 envs.** The "essentially free" result in
   `notes/pusht-detour.md` was measured on `physx_cuda`. This pipeline is
   `physx_cpu`. Smoke gate 4 prints ms/step; measure before assuming.

---

## T-II logging schema

Decide this once, here, and never re-derive it. Angle-wrap conventions are what
make a failure characterisation incoherent.

Per episode, log:

```
{seed, env_idx, cubeA_x, cubeA_y, cubeA_theta,
                cubeB_x, cubeB_y, cubeB_theta,
 separation, relative_yaw, success, ep_len}
```

- `separation` is XY Euclidean distance between cube centres (metres).
- `relative_yaw` is `cubeB_theta − cubeA_theta`.
- **All angles wrapped to `(−π, π]`.** Both the absolute thetas and the relative
  yaw. Wrap once at log time, never at analysis time.

`separation` and `relative_yaw` are the hypothesised failure axes: too close and
the grasp approach on A collides with B; too far and the place phase runs out of
steps. Success-rate-versus-separation is the curve T-II is looking for.

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
  error doesn't say so. Run `bash transfer.sh check` first thing on every pod.
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
```

---

## The scripts

Four files, each with one job:

| | |
|---|---|
| `setup_runpod.sh` | Builds a fresh pod: packages, Vulkan/EGL ICDs, venv, ManiSkill clone. Writes `env.sh`. Deliberately checks nothing at runtime. |
| `smoke_test.sh` | Four gates, ~3 min. Sim constructs; renderer emits real pixels; dataset action dim matches the env; seeds determine initial states. Gate 4 is T-II's prerequisite. |
| `run_pipeline.sh` | The recipe. `data` \| `train` \| `train-rgb` \| `all`. |
| `transfer.sh` | `check` \| `info` \| `send`. Run `check` before anything else on a new pod. |

`ENV_ID` is a variable in both `run_pipeline.sh` and `smoke_test.sh`. The task has
changed once; changing it again should cost one export, not a rename.

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
  startup; `run_pipeline.sh` checks.
- At high `--num_eval_envs` the logged scalar is a single mean over the batch, so
  per-eval variance is not recoverable from wandb. **T-I error bars must come
  from a standalone harness, not the training logs.**

---

## Known traps

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

---

## Current state

*Volatile — update as work lands.*

- Repo retargeted at StackCube; PushT scripts deleted, its record kept in
  `notes/pusht-detour.md`.
- **Nothing has been run on StackCube.** No demos, no datasets, no policy, no
  measurements. Every number in this repo is from PushT and belongs only to the
  detour note.
- The four open questions above are the first thing to resolve on the pod.
- Then: `bash smoke_test.sh` (gates 1, 2, 4 pass without a dataset; 3 skips),
  `bash run_pipeline.sh data`, `bash smoke_test.sh` again for gate 3, then the
  state training run.

**The gate on everything downstream** is a state-obs policy with success
meaningfully above 0 *and* meaningfully below 1. Above 0 is what PushT never
delivered. Below 1 is what PushCube would not have delivered — if StackCube
saturates, T-II has nothing to characterise and the task choice needs revisiting
immediately, not after the RGB run.

The RGB run's checkpoint is the T-I deliverable and the frozen base for T-IV.
Failure modes must be characterised on the *visual* policy, since that is what
T-III and T-IV improve.

**Decision point:** commit to full T-IV on **one** failure mode. One mode with
clean curves, proper 3-seed eval on both distributions, and honest analysis beats
two half-trained runs. State the choice explicitly in the report.

**Never cut:** the T-I baseline number, the quantitative failure
characterisation, and the nominal-distribution re-evaluation after finetuning.
Those three are the spine of the submission.
