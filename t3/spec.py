#!/usr/bin/env python
"""The contract the LLM-generated code must satisfy. Pure stdlib.

T-III asks an LLM for two pieces of executable Python: a dense reward function
and an episode-configuration sampler. This file is the single source of truth
for what "acceptable" means, and it has three consumers that cannot be allowed
to drift apart:

    1. the PROMPT     t3/assemble.py renders contract_markdown() and
                      api_surface_markdown() into the text the model reads;
    2. the LOADER     t3/loader.py walks the AST against ALLOWED_IMPORTS and the
                      required signatures;
    3. the SUMMARY    t3/summary.py reads the thresholds below and prints them
                      against what t3/check.py measured.

Imports `math` and nothing else, so the definition of "acceptable" and its tests
run on a laptop with the bare system interpreter. Same rule, same reason, as
t2/geometry.py.

THRESHOLDS HERE ARE ADVISORY. t3/summary.py prints WARN against them and exits 0
regardless. That is deliberate: this harness used to be a gate that refused a
generation over a probe ordering, which costs a regeneration cycle and blocks
T-IV. A number below threshold is a finding for the report. See t3/README.md.
"""
import math

# ---------------------------------------------------------------------------
# what the LLM must emit
# ---------------------------------------------------------------------------
#
# Parameter names are checked EXACTLY and IN ORDER, not merely by arity: a
# generation that emits compute_reward(env, action, obs, info) is syntactically
# fine, imports correctly, and silently computes the reward from the wrong
# tensor. That has to be a load error, not a debugging session.

REWARD_FILE = "reward.py"
SAMPLER_FILE = "sampler.py"

REWARD_MODULE_REQUIRES = {
    "compute_reward": ("env", "obs", "action", "info"),
}
SAMPLER_MODULE_REQUIRES = {
    "sample_cube_poses": ("b", "device"),
}
REWARD_MAX_NAME = "REWARD_MAX"

ALLOWED_IMPORTS = frozenset({"torch", "math"})
MAX_SOURCE_LINES = 400

# Bare names that must never be loaded: the exec escape surface.
FORBIDDEN_NAMES = frozenset({
    "eval", "exec", "compile", "open", "__import__", "globals", "locals",
    "getattr", "setattr", "delattr", "vars", "input", "breakpoint",
})

# Attribute names that mutate the simulator from inside a reward. The nastiest
# hack available here is a reward that moves cubeA onto cubeB itself and then
# collects the bonus for cubeA being on cubeB.
FORBIDDEN_ATTRS = frozenset({
    "set_pose", "set_state_dict", "set_state", "set_qpos", "set_qvel",
    "step", "reset",
})

# Warned about, not rejected: these force a GPU->CPU sync every step, which is a
# performance problem rather than a correctness one.
DISCOURAGED_ATTRS = frozenset({"item", "cpu", "numpy", "tolist"})

# ---------------------------------------------------------------------------
# the API surface - DOCUMENTATION, not enforcement
# ---------------------------------------------------------------------------
#
# These two sets render into the prompt so the model knows exactly what it may
# read, which is most of their value. The loader warns when generated code goes
# outside them and does not reject: an attribute whitelist is the likeliest way
# a perfectly good reward gets refused, and reading something unexpected is a
# thing to notice rather than a thing to block.

ALLOWED_ENV_ATTRS = frozenset({
    "num_envs", "device", "cube_half_size",
    "cubeA.pose.p", "cubeA.pose.q", "cubeA.pose.raw_pose",
    "cubeA.linear_velocity", "cubeA.angular_velocity",
    "cubeB.pose.p", "cubeB.pose.q", "cubeB.pose.raw_pose",
    "cubeB.linear_velocity", "cubeB.angular_velocity",
    "agent.tcp.pose.p", "agent.tcp.pose.q", "agent.tcp.pose.raw_pose",
    "agent.robot.get_qpos", "agent.robot.get_qvel", "agent.robot.get_qlimits",
})

# The roots of the above, for loader.py's cheap root-only check.
ALLOWED_ENV_ROOTS = frozenset(a.split(".")[0] for a in ALLOWED_ENV_ATTRS)

# The four keys StackCubeEnv.evaluate() returns (stack_cube.py:112-131).
ALLOWED_INFO_KEYS = frozenset({
    "is_cubeA_grasped", "is_cubeA_on_cubeB", "is_cubeA_static", "success",
})

# ---------------------------------------------------------------------------
# the physics the sampler must respect
# ---------------------------------------------------------------------------
#
# Read out of StackCube's own _initialize_episode (stack_cube.py:78-110) rather
# than chosen. A sampler that leaves this envelope is not "biased toward the
# failure regime", it is sampling states the base policy has never seen - and
# T-IV's residual would then be scored on a distribution shift rather than on a
# failure mode.

CUBE_HALF = 0.02                     # stack_cube.py:64,72 - 40 mm cubes
CUBE_Z = 0.02                        # stack_cube.py:82 - xyz[:, 2] = 0.02

# stack_cube.py:86 region [[-0.1,-0.2],[0.1,0.2]] plus the SHARED offset at :85
# (rand*0.2 - 0.1) that both cubes get. The support is the sum of the two.
SUPPORT_X = (-0.2, 0.2)
SUPPORT_Y = (-0.3, 0.3)

# stack_cube.py:88-90: radius = ||[0.02, 0.02]|| + 0.001, and
# UniformPlacementSampler rejects at fixtures_radii + radius (samplers.py:63),
# so the enforced floor on CENTRE separation is twice that.
MIN_SEPARATION = 2 * (math.dist((0.0, 0.0), (0.02, 0.02)) + 0.001)   # 0.05857

# DERIVED FROM THE SUPPORT, never assumed. This was hard-coded at 0.80 on the
# reasoning that the IK saturates beyond it, and that was wrong twice over:
#
#   - The environment's own sampler produces distances up to 0.8515 over 25,000
#     seeds (1.0% of placements exceed 0.80), so 0.80 does not describe the
#     support it claimed to describe.
#   - 19.5% of the measured `farb` episodes have dist_B > 0.80. The contract
#     therefore forbade the model from sampling a fifth of the very region it
#     was being asked to target, and check.py's `reach_bad` counted a correct
#     sampler's rows as invalid. Both sides of the same wrong number.
#
# The real bound is the corner of the support box, which is where the
# environment can actually place a cube. Anything the env can produce, the
# sampler may produce; that is the whole of the constraint.
PANDA_BASE = (-0.615, 0.0)
REACH_MAX = math.dist(PANDA_BASE, (SUPPORT_X[1], SUPPORT_Y[1]))      # 0.86846

# ---------------------------------------------------------------------------
# thresholds - advisory, see the module docstring
# ---------------------------------------------------------------------------
#
# All of the alignment thresholds are SCALE-FREE - AUCs and ratios, never a
# reward value. The LLM picks its own scale and REWARD_MAX may be 1.0 or 100.0;
# a threshold on an absolute return would need retuning per generation.

ALIGN_N_TARGET = 100
ALIGN_AUC_MIN = 0.75                 # cumulative reward vs success_once
ALIGN_Z_MIN = 3.0                    # z against the null

# The margin, as a fraction of the observed return range, by which each stage's
# mean return must exceed the previous stage's:
#
#     "grasped, not placed"  <  "placed, not success"  <  "success"
#
# THIS IS THE STAGE TEST, AND A STAGE-CONDITIONAL AUC IS NOT - which is worth
# stating plainly, because the opposite was assumed while designing this file
# and the arithmetic says otherwise.
#
# A conditional AUC for `gap` would be AUC(return ; ever_placed | ever_grasped),
# and the successful episodes are a SUBSET of the placed ones. So a reward that
# ranks success at the top - which the unconditional check already requires -
# wins every conditional pair for free. At T-II's measured stage rates for `gap`
# (96 grasped, 66 placed, 52 successful of 100) that floor is
#
#     52 * 30 / (66 * 30) = 0.788
#
# already above the 0.70 it used to be gated at. A reward paying nothing
# whatsoever for placement would have passed it.
#
# What DOES catch that reward is requiring the mean return to rise from one
# stage to the next with a real margin. A grasp-farming reward inverts the first
# step however large its success bonus. Do not "restore" the conditional AUC as
# the stage test.
ALIGN_STAGE_GAP_FRAC = 0.05

SAMPLER_DRAWS = 4096
SAMPLER_HIT_RATE_MIN = 0.60          # fraction of draws inside geometry.MODES[tag]
SAMPLER_ENRICHMENT_MIN = 10.0        # hit_rate / base_rate, base rate from the index
SAMPLER_SD_MIN_XY = 0.015            # m, per coordinate - a collapsed sampler
                                     # hits the region perfectly and teaches
                                     # T-IV one initial state

# ---------------------------------------------------------------------------
# the CSV schemas
# ---------------------------------------------------------------------------
#
# Declared beside the thresholds that read them, for the reason
# geometry.COLUMNS gives: the CSV is the contract between the half of this
# harness that needs a simulator and the half that runs on a laptop.

ALIGN_COLUMNS = [
    "run_id", "mode", "arm", "policy_seed", "seed",
    "cubeA_x", "cubeA_y", "cubeA_theta", "cubeB_x", "cubeB_y", "cubeB_theta",
    "face_gap", "dist_A", "dist_B", "dist_max", "dist_min",
    "ep_return", "ep_reward_mean", "ep_len",
    "success_once", "success_at_end",
    "ever_grasped", "ever_placed", "ever_static",
]

SAMPLER_COLUMNS = [
    "i", "cubeA_x", "cubeA_y", "cubeA_theta", "cubeB_x", "cubeB_y",
    "cubeB_theta", "separation", "face_gap", "dist_A", "dist_B", "in_region",
]

# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def auc(scores, labels):
    """AUC of `scores` as a classifier of binary `labels`, tie-corrected.

    The Mann-Whitney U statistic over n1*n0: "the probability a randomly chosen
    positive scores above a randomly chosen negative", ties counting half.
    Returns nan when either class is empty - the geometry.wilson(0, 0)
    precedent, a statistic with no data returns nan rather than a number that
    reads as a measurement.
    """
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0:
        return float("nan")
    vals = pos + neg
    order = sorted(range(n1 + n0), key=vals.__getitem__)
    ranks = [0.0] * (n1 + n0)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return (sum(ranks[:n1]) - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def se_null_auc(n1, n0):
    """SD of the AUC under the null that scores carry no information.

    Printed beside every AUC so a reader can see whether the test had the power
    to reject anything - the job geometry.wilson does for every rate in T-II.
    """
    if n1 <= 0 or n0 <= 0:
        return float("nan")
    return math.sqrt((n1 + n0 + 1) / (12.0 * n1 * n0))


def auc_z(a, n1, n0):
    """Standard deviations of separation between an observed AUC and chance."""
    se = se_null_auc(n1, n0)
    if not (se > 0) or a != a:
        return float("nan")
    return (a - 0.5) / se


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def sd(xs):
    xs = list(xs)
    if len(xs) < 2:
        return float("nan")
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


# ---------------------------------------------------------------------------
# rendering the contract into the prompt
# ---------------------------------------------------------------------------
#
# The prompt is assembled from files under t3/prompts/, but the parts that MUST
# match the checker are generated from the constants above and spliced in at
# {{CONTRACT}} and {{API_SURFACE}}. Writing them by hand in the prompt file is
# how a checker and a prompt come to disagree, and the model gets blamed.


def _sig(name, params):
    return f"def {name}({', '.join(params)}):"


def contract_markdown():
    """The exact signatures and rules, rendered for the prompt."""
    r = _sig("compute_reward", REWARD_MODULE_REQUIRES["compute_reward"])
    s = _sig("sample_cube_poses", SAMPLER_MODULE_REQUIRES["sample_cube_poses"])
    return f"""\
### `{REWARD_FILE}` — the dense reward

```python
{REWARD_MAX_NAME} = 8.0            # module-level, a positive float literal

{r}
    \"\"\"-> torch.Tensor, shape (env.num_envs,), floating dtype, on env.device.\"\"\"
```

Hard requirements:

* Every returned value lies in `[0, {REWARD_MAX_NAME}]`. The env divides by
  `{REWARD_MAX_NAME}` to produce a normalised reward in `[0, 1]` for PPO.
* It is a **pure function of the current simulator state**. No randomness, no
  module-level counters, no state carried between calls. It is called once per
  `env.step()` for every parallel environment at once, and calling it twice on
  the same state must return bitwise identical values.
* It **must not mutate anything** reachable from `env`. A reward that moves a
  cube and then collects the bonus for the cube being moved is rejected.
* Fully batched. No `if` on a tensor, no `.item()`, `.cpu()`, `.numpy()` or
  `.tolist()` — those force a GPU→CPU sync on every step of training. Use
  `torch.where`, boolean masks, and in-place masked assignment.
* Parameter names must be exactly `{', '.join(REWARD_MODULE_REQUIRES['compute_reward'])}`, in that order.

### `{SAMPLER_FILE}` — the episode-configuration sampler

```python
{s}
    \"\"\"-> dict with exactly these four keys, torch.float32 on `device`:

           cubeA_xyz  (b, 3)   z must equal {CUBE_Z} exactly
           cubeA_quat (b, 4)   wxyz, unit norm, rotation about z ONLY (x = y = 0)
           cubeB_xyz  (b, 3)
           cubeB_quat (b, 4)
    \"\"\"
```

Hard requirements:

* Draw randomness with `torch.rand` / `torch.randn` / `torch.randint` **only**.
  The caller has already entered `with torch.device(device)`, so a bare
  `torch.rand((b, 2))` lands on the right device — exactly as the environment's
  own `_initialize_episode` does it. `random` and `numpy.random` are forbidden:
  the environment seeds `torch`'s generator and nothing else, so drawing from
  anywhere else silently destroys `reset(seed=s)` reproducibility.
* Both cubes lie inside the nominal support: `x ∈ [{SUPPORT_X[0]}, {SUPPORT_X[1]}]`,
  `y ∈ [{SUPPORT_Y[0]}, {SUPPORT_Y[1]}]`, `z = {CUBE_Z}`. The biased distribution must be a
  **subset** of the distribution the base policy was trained on — otherwise the
  residual is being asked to handle a distribution shift rather than a failure.
* Centre-to-centre separation ≥ `{MIN_SEPARATION:.5f}` m for every row. That is the
  environment's own rejection-sampling floor for 40 mm cubes; below it the cubes
  physically overlap.
* Both cubes within `{REACH_MAX:.3f}` m of the Panda base at `(-0.615, 0.0)`. That
  is the far corner of the support box above, not an extra constraint — it is
  simply how far the environment itself can place a cube. **Do not back off from
  it.** Where the arm struggles is a property of the failure mode you were given,
  not of this bound, and treating the bound as a wall to stay clear of will put
  your region in the wrong place.
* Rejection sampling must use a **bounded `for`**, never a `while`. Fall back to
  the last valid draw if the budget runs out; never return an invalid row.
* The draws must be **varied**. A sampler that returns one configuration hits
  the target region 100% of the time and teaches the policy nothing that
  generalises.
* Parameter names must be exactly `{', '.join(SAMPLER_MODULE_REQUIRES['sample_cube_poses'])}`, in that order.

### Both files

* Import **only** `torch` and `math`, at module level. Nothing else — no
  `numpy`, no `os`, no `random`.
* No classes, no decorators, no `while` loops, no `global`.
* Under {MAX_SOURCE_LINES} lines each.
* Self-contained: the two files do not import each other.
"""


def api_surface_markdown():
    """The attribute surface the generated code should read."""
    env_attrs = "\n".join(f"    env.{a}" for a in sorted(ALLOWED_ENV_ATTRS))
    info_keys = "\n".join(f'    info["{k}"]' for k in sorted(ALLOWED_INFO_KEYS))
    return f"""\
These are what the generated code should read. Reading anything else is
flagged for review, so stay inside this surface unless you have a reason.

```
{env_attrs}
```

```
{info_keys}
```

Notes on what these are:

* `.pose.p` is `(num_envs, 3)` position; `.pose.q` is `(num_envs, 4)` wxyz
  quaternion; `.pose.raw_pose` is `(num_envs, 7)`, position then quaternion.
* `env.cube_half_size` is a length-3 tensor, every element {CUBE_HALF} — the cubes are
  {CUBE_HALF * 2000:.0f} mm.
* `env.agent.robot.get_qpos()` is `(num_envs, 9)` for the Panda; the last two
  entries are the two finger joints, and their sum over the gripper width is how
  the built-in reward measures "has the gripper opened".
* Every `info[...]` value is a **boolean tensor of shape (num_envs,)**, not a
  Python bool. Use them as masks.
* `obs` and `action` are passed for completeness. `obs` is whatever the current
  observation mode produces and **does not contain the cube poses under the RGB
  observation mode this project uses** — read poses from `env`, never from `obs`.
"""
