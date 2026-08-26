#!/usr/bin/env python
"""The contract the LLM-generated code must satisfy, and the thresholds that
decide whether it did. Pure stdlib.

T-III asks an LLM to write two pieces of executable Python: a dense reward
function and an episode-configuration sampler. Both are then run inside
ManiSkill, so "is this artifact acceptable" has to be answerable BEFORE any GPU
minute is spent, and answerable again afterwards from files alone.

This file is that answer's single source of truth. It has three consumers, and
the whole point is that they cannot drift apart:

    1. the PROMPT       - t3/assemble.py renders contract_markdown() and
                          api_surface_markdown() into the text the model reads,
                          so the model is told exactly what the checker checks;
    2. the LOADER       - t3/loader.py walks the AST against ALLOWED_IMPORTS,
                          ALLOWED_ENV_ATTRS and the required signatures;
    3. the GATE         - t3/verify.py reads the thresholds below and applies
                          them to the CSVs the measurement scripts wrote.

It imports `math` and nothing else - no numpy, no torch, no gymnasium - so the
definition of "acceptable", its tests, and the gate all run on a laptop with the
bare system interpreter. Same rule, and the same reason, as t2/geometry.py: this
is the layer where being wrong is most expensive, so feedback here should be
free.

See t3/README.md for the method this sits inside.
"""
import math

# ---------------------------------------------------------------------------
# what the LLM must emit
# ---------------------------------------------------------------------------
#
# Two modules, two required functions, one required constant. Parameter names
# are checked EXACTLY and IN ORDER, not merely by arity: a generation that emits
# compute_reward(env, action, obs, info) is syntactically fine, imports
# correctly, and silently computes a reward from the wrong tensor. That has to
# be a load error, not a debugging session.

REWARD_FILE = "reward.py"
SAMPLER_FILE = "sampler.py"

REWARD_MODULE_REQUIRES = {
    "compute_reward": ("env", "obs", "action", "info"),
}
SAMPLER_MODULE_REQUIRES = {
    "sample_cube_poses": ("b", "device"),
}
REWARD_MAX_NAME = "REWARD_MAX"

# The four keys StackCubeEnv.evaluate() returns (stack_cube.py:112-131). The
# reward may read these and nothing else out of `info` - a literal subscript, so
# a typo is a load error rather than a KeyError at step 40,000 of training.
ALLOWED_INFO_KEYS = frozenset({
    "is_cubeA_grasped", "is_cubeA_on_cubeB", "is_cubeA_static", "success",
})

# Attribute chains rooted at the `env` parameter. Prefix match: "cubeA.pose.p"
# admits env.cubeA.pose.p and nothing shorter. Deliberately small - every entry
# is state the reward can legitimately want, and the omissions are the point:
#
#   env.evaluate()        would recompute the flags that are already in `info`,
#                         at the cost of a second contact query per step;
#   env.get_state_dict()  is the mutation surface, and layer B snapshots it;
#   env.elapsed_steps     makes the reward non-stationary - "reward for
#                         surviving" is a real hack and a time-dependent term is
#                         how it gets written;
#   env.scene, env._*     no legitimate use, and both are escape hatches.
ALLOWED_ENV_ATTRS = frozenset({
    "num_envs", "device", "cube_half_size",
    "cubeA.pose.p", "cubeA.pose.q", "cubeA.pose.raw_pose",
    "cubeA.linear_velocity", "cubeA.angular_velocity",
    "cubeB.pose.p", "cubeB.pose.q", "cubeB.pose.raw_pose",
    "cubeB.linear_velocity", "cubeB.angular_velocity",
    "agent.tcp.pose.p", "agent.tcp.pose.q", "agent.tcp.pose.raw_pose",
    "agent.robot.get_qpos", "agent.robot.get_qvel", "agent.robot.get_qlimits",
})

ALLOWED_IMPORTS = frozenset({"torch", "math"})

# Bare names that must never be loaded. The first group is the exec escape
# surface; the second is mis-specification rather than malice.
FORBIDDEN_NAMES = frozenset({
    "eval", "exec", "compile", "open", "__import__", "globals", "locals",
    "getattr", "setattr", "delattr", "vars", "input", "breakpoint",
    "exit", "quit", "help", "memoryview",
})

# Attribute names that must never be accessed anywhere in the module.
#
#   item/cpu/numpy/tolist  force a GPU->CPU sync every step. Under PPO at batch
#                          512 on physx_cuda this alone can dominate wall clock,
#                          and it is invisible at the num_envs=1 width a naive
#                          test would use.
#   set_*/step/reset       mutate the simulator from inside a reward. Layer B
#                          catches it with a state-dict snapshot; this catches
#                          it in 0.2 s without a simulator.
FORBIDDEN_ATTRS = frozenset({
    "item", "cpu", "numpy", "tolist", "detach_",
    "set_pose", "set_state_dict", "set_state", "set_qpos", "set_qvel",
    "step", "reset", "evaluate", "get_state_dict", "get_state",
})

MAX_SOURCE_LINES = 400
MAX_AST_NODES = 4000

# ---------------------------------------------------------------------------
# the physics the sampler must respect
# ---------------------------------------------------------------------------
#
# Every constant here is read out of StackCube's own _initialize_episode
# (stack_cube.py:78-110) rather than chosen. A sampler that leaves this envelope
# is not "biased toward the failure regime", it is sampling states the base
# policy has never seen - and T-IV's residual would then be scored on a
# distribution shift rather than on a failure mode.

CUBE_HALF = 0.02                     # stack_cube.py:64,72 - 40 mm cubes
CUBE_Z = 0.02                        # stack_cube.py:82 - xyz[:, 2] = 0.02

# stack_cube.py:86 - region = [[-0.1, -0.2], [0.1, 0.2]], plus the SHARED
# offset xy = rand*0.2 - 0.1 at :85 which both cubes get. The reachable support
# is therefore the sum of the two.
TABLE_REGION = ((-0.1, -0.2), (0.1, 0.2))
SHARED_OFFSET = 0.1                  # |offset| <= this, per axis
SUPPORT_X = (-0.2, 0.2)
SUPPORT_Y = (-0.3, 0.3)

# stack_cube.py:88-90: radius = ||[0.02, 0.02]|| + 0.001, and
# UniformPlacementSampler rejects at fixtures_radii + radius (samplers.py:63),
# so the enforced floor on CENTRE separation is twice that.
MIN_SEPARATION = 2 * (math.dist((0.0, 0.0), (0.02, 0.02)) + 0.001)   # 0.05857

# Above this the IK saturates and no bounded residual recovers a target the arm
# cannot reach: T-II measured grasp 0.815 above 740 mm (CLAUDE.md, the
# both-cubes-far corner). Not 0.76 - mode `farb` deliberately wants
# dist_B >= 0.76, so the ceiling has to sit above the region it is guarding.
REACH_MAX = 0.80

# ---------------------------------------------------------------------------
# which stage each mode fails at
# ---------------------------------------------------------------------------
#
# This is what makes layer D an alignment test rather than a sanity check. T-II
# measured that the two modes break at DIFFERENT stages:
#
#            grasp   place   hold|place
#   nominal  0.984   0.884   0.807
#   gap      0.955   0.659   0.793      <- placement collapses
#   farb     1.000   0.854   0.657      <- holding collapses
#
# A reward that farms reach-and-grasp shaping scores a high UNCONDITIONAL AUC,
# because grasping correlates with everything downstream. Conditioning on the
# stage the mode actually fails at is what separates a useful reward from a
# plausible-looking one, so each mode names its own conditional test.
MODE_STAGE = {
    # given the cube was grasped, does the reward rank the episodes that
    # actually got it placed above the ones that did not?
    "gap":  dict(given="ever_grasped", target="ever_placed",
                 label="placement | grasped"),
    # given it was placed, does the reward rank the episodes where the stack
    # stayed above the ones where it did not?
    "farb": dict(given="ever_placed", target="success_once",
                 label="success | placed"),
    "nominal": dict(given="ever_grasped", target="ever_placed",
                    label="placement | grasped"),
}

# ---------------------------------------------------------------------------
# the probe battery - layer C
# ---------------------------------------------------------------------------
#
# Hand-built states, scored by the generated reward and by evaluate(). They cost
# ~90 s and they catch the mis-specifications a rollout cannot: the base policy
# never puts cubeB on cubeA, so no number of rollouts reveals a reward whose
# goal argument is swapped.
#
# Every state here is constructed by set_pose on a single-process env. NOTE the
# limit that follows from that, because it cost an afternoon to find:
# Panda.is_grasping reads pairwise CONTACT FORCES from the last physics step
# (panda.py:225-253 -> scene.py:821-833). After a reset plus set_pose with no
# step there are no contacts, so is_cubeA_grasped is False in every injected
# state. P7_held therefore cannot be injected - it is restored from a cached
# state dict captured during a real rollout and then stepped twice with the
# gripper closing, and probes.py asserts the flag actually came back.

PROBES = (
    ("P0_success",  "cubeA stacked on cubeB, released, static. The reference - "
                    "evaluate() must return success=True here."),
    ("P1_hover",    "cubeA 120 mm above the goal. Above it, not on it."),
    ("P2_adjacent", "cubeA on the table 50 mm from cubeB. Beside, not stacked."),
    ("P3_knocked",  "cubeB shoved to the table edge, cubeA at its reset pose. "
                    "Did the reward pay for destroying the target?"),
    ("P4_inverted", "cubeB stacked on cubeA - the goal arguments swapped."),
    ("P5_offtable", "cubeA below the table at the goal xy. Catches a reward "
                    "that measures xy distance and forgets z."),
    ("P6_far",      "cubeA 300 mm from cubeB on the table. The far end of the "
                    "shaping ramp."),
    ("P7_held",     "cubeA grasped and held above cubeB, never released. "
                    "Restored from a cached rollout state, not injected."),
    ("P8_start",    "the untouched reset state."),
)

# (high, low, margin as a fraction of REWARD_MAX). Strict `>` alone is not
# enough: a reward that returns 8*success ties every non-success probe at zero
# and would pass on a technicality. The margin is what makes the ordering a
# statement about shaping rather than about float noise.
PROBE_ORDERINGS = (
    ("P0_success", "P1_hover",    0.05),
    ("P0_success", "P2_adjacent", 0.05),
    ("P0_success", "P3_knocked",  0.05),
    ("P0_success", "P5_offtable", 0.05),
    ("P0_success", "P6_far",      0.05),
    ("P0_success", "P8_start",    0.05),
    # The canonical mis-specification gets a wider margin: a swapped goal is not
    # a near-miss, it is the wrong task.
    ("P0_success", "P4_inverted", 0.10),
    # THE ordering. "Grab it and hold it up forever" is what a grasp-and-place
    # reward incentivises, and StackCube's success criterion explicitly requires
    # the release (stack_cube.py:126 - success = on_cubeB & static & ~grasped).
    # The stock reward has a whole ungrasp term for this (stack_cube.py:168).
    ("P0_success", "P7_held",     0.05),
)

# Monotone sweeps: 24 states each, cubeA walked away from the goal along one
# axis at a time. Two assertions, and the second is the one that earns its keep:
#
#   MONOTONE - the reward must not go UP as the cube moves away from the goal.
#              Two violations are tolerated; contact jitter produces about one.
#   RANGE    - the reward must actually MOVE across the sweep. A reward that
#              measures only xy distance is perfectly monotone along the z sweep
#              because it is CONSTANT there, and would pass a monotonicity test
#              while being blind to the axis stacking happens along. Requiring a
#              real range on both sweeps is what catches it.
#
# This replaces a probe pair (cubeA beside the target vs cubeA below the floor)
# that was in an earlier draft of this file. That comparison turned out to be
# decided by the parked arm's distance to the cube rather than by anything about
# stacking, so the stock reward itself passed or failed it by luck - and a gate
# the calibration arm fails at random is worse than no gate.
SWEEP_STEPS = 24
SWEEP_MAX_VIOLATIONS = 2
SWEEP_RANGE_MIN = 0.02               # (max - min) / REWARD_MAX, per sweep
SWEEP_Z = (0.06, 0.30)               # cubeA z at the goal xy
SWEEP_R = (0.0, 0.20)                # radial xy offset at the goal z

# ---------------------------------------------------------------------------
# thresholds - layer D (rollout alignment) and layer E (sampler)
# ---------------------------------------------------------------------------
#
# All of layer D's thresholds are SCALE-FREE - AUCs and ratios, never a reward
# value. The LLM picks its own scale, and REWARD_MAX may be 1.0 or 100.0; a gate
# that compared absolute returns would have to be retuned for every generation.

ALIGN_N_MIN = 60                     # below this the test has no power - see below
ALIGN_N_TARGET = 100                 # what run.sh runs
ALIGN_AUC_MIN = 0.75                 # cumulative reward vs success_once
ALIGN_Z_MIN = 3.0                    # z against the null, so a lucky split cannot pass
ALIGN_COND_AUC_MIN = 0.70            # the stage-conditional AUC
ALIGN_COND_N_MIN = 25                # a conditional stratum smaller than this is reported, not gated
ALIGN_MEAN_GAP_FRAC = 0.10           # mean(success) - mean(fail), in units of the observed range
ALIGN_CONTROL_RATIO = 1.25           # policy vs jitter, and policy vs zero

# The margin, as a fraction of the observed return range, by which each stage's
# mean return must exceed the previous stage's.
#
# THIS IS THE BINDING STAGE TEST, and the conditional AUC above is not - which
# is worth stating plainly because the opposite was assumed while designing this
# file, and the arithmetic says otherwise.
#
# The conditional AUC for `gap` is AUC(return ; ever_placed | ever_grasped), and
# the successful episodes are a SUBSET of the placed ones. So if a reward ranks
# success at the top - which the unconditional check already requires - those
# episodes win every conditional pair for free. At T-II's measured stage rates
# (96 grasped, 66 placed, 52 successful of 100) that floor is
#
#     52 * 30 / (66 * 30) = 0.788
#
# already above the 0.70 threshold. A reward that pays nothing whatsoever for
# placement, so long as it pays for success, would pass it. The conditional AUC
# is kept because it is a graded number worth reporting and because it does
# catch gross failure, but it cannot be the check the gate leans on.
#
# What DOES catch that reward is requiring the mean return to rise from one
# stage to the next with a real margin: "grasped but never placed" < "placed but
# it did not stay" < "succeeded". A grasp-farming reward inverts the first of
# those, and no amount of success bonus hides it.
ALIGN_STAGE_GAP_FRAC = 0.05

# Layer E.
SAMPLER_DRAWS = 4096
SAMPLER_HIT_RATE_MIN = 0.60          # fraction of draws inside geometry.MODES[tag]
SAMPLER_ENRICHMENT_MIN = 10.0        # hit_rate / base_rate; base_rate computed from the index
SAMPLER_SD_MIN_XY = 0.015            # m, per coordinate - a collapsed sampler overfits T-IV
SAMPLER_SD_MIN_GAP = 0.004           # m, face_gap
SAMPLER_YAW_BINS = 12                # 30-degree bins
SAMPLER_YAW_BINS_MIN = 8             # of 12, per cube
SAMPLER_DISTINCT_FRAC = 0.90         # distinct poses at 1 mm rounding
SAMPLER_COVER_TOL = 0.020            # m - a draw "covers" an eval seed within this
SAMPLER_COVER_FRAC = 0.80            # fraction of the T-II eval seeds that must be covered

# ---------------------------------------------------------------------------
# the CSV schemas
# ---------------------------------------------------------------------------
#
# Declared here with the thresholds that read them, for the reason
# geometry.COLUMNS gives: the CSV is the contract between the half of this
# harness that needs a simulator and the half that must run on a laptop. A
# schema declared in the writer is a schema the readers discover by failing.

PROBE_COLUMNS = [
    "probe", "reward", "reward_norm",
    "is_cubeA_grasped", "is_cubeA_on_cubeB", "is_cubeA_static", "success",
    "cubeA_x", "cubeA_y", "cubeA_z", "cubeB_x", "cubeB_y", "cubeB_z",
]

SWEEP_COLUMNS = ["sweep", "i", "offset", "reward"]

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
#
# Here rather than in a helper module because a threshold and the statistic it
# thresholds should be readable together - the same reason geometry.wilson sits
# beside geometry.DISCOVERY. Pure stdlib, so t3/test_spec.py checks them against
# hand-computed cases in a second.


def auc(scores, labels):
    """AUC of `scores` as a classifier of binary `labels`, tie-corrected.

    This is the Mann-Whitney U statistic divided by n1*n0, which is exactly
    "the probability that a randomly chosen positive scores above a randomly
    chosen negative", with ties counting half. Returns nan when either class is
    empty - the geometry.wilson(0, 0) precedent: a statistic with no data
    returns nan rather than a number that reads as a measurement.
    """
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0:
        return float("nan")
    vals = pos + neg
    order = sorted(range(n1 + n0), key=vals.__getitem__)
    # average ranks over ties
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
    r1 = sum(ranks[:n1])
    return (r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def se_null_auc(n1, n0):
    """SD of the AUC under the null that scores carry no information.

    sqrt((n1 + n0 + 1) / (12 * n1 * n0)). Printed beside every AUC so a reader
    can see whether the test had the power to reject anything - the same job
    geometry.wilson does for every rate in T-II.
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


def conditional_auc(scores, labels, given):
    """AUC restricted to the rows where `given` is true. -> (auc, n1, n0).

    The stage-conditional test. `given` selects the stratum the mode's failure
    lives in; within it, `labels` is the stage that actually collapses.
    """
    s = [x for x, g in zip(scores, given) if g]
    y = [x for x, g in zip(labels, given) if g]
    return auc(s, y), sum(1 for v in y if v), sum(1 for v in y if not v)


def point_biserial(scores, labels):
    """Pearson r between a continuous score and a 0/1 label.

    Reported alongside the AUC because it is the number a reader recognises,
    but NOT gated on: r is sensitive to the shape of the score distribution and
    the AUC is not, and the LLM chooses that shape.
    """
    n = len(scores)
    if n < 2:
        return float("nan")
    m = sum(scores) / n
    sd = math.sqrt(sum((s - m) ** 2 for s in scores) / n)
    if sd == 0:
        return float("nan")
    n1 = sum(1 for y in labels if y)
    n0 = n - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    m1 = sum(s for s, y in zip(scores, labels) if y) / n1
    m0 = sum(s for s, y in zip(scores, labels) if not y) / n0
    return (m1 - m0) / sd * math.sqrt(n1 * n0 / float(n * n))


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
# how a checker and a prompt come to disagree, and the model gets blamed for it.


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
  `env.step()` for every parallel environment at once — batch width 512 on GPU
  during training and width 1 on CPU during evaluation — and calling it twice on
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
  anywhere else silently destroys `reset(seed=s)` reproducibility, and every
  evaluation in this project addresses episodes by seed.
* Both cubes lie inside the nominal support: `x ∈ [{SUPPORT_X[0]}, {SUPPORT_X[1]}]`,
  `y ∈ [{SUPPORT_Y[0]}, {SUPPORT_Y[1]}]`, `z = {CUBE_Z}`. The biased distribution must be a
  **subset** of the distribution the base policy was trained on — otherwise the
  residual is being asked to handle a distribution shift rather than a failure.
* Centre-to-centre separation ≥ `{MIN_SEPARATION:.5f}` m for every row. That is the
  environment's own rejection-sampling floor for 40 mm cubes; below it the cubes
  physically overlap.
* Both cubes within {REACH_MAX} m of the Panda base at `(-0.615, 0.0)`. Beyond that
  the arm's IK saturates and no bounded residual can help.
* Rejection sampling must use a **bounded `for`**, never a `while`. Fall back to
  the last valid draw if the budget runs out; never return an invalid row.
* The draws must be **varied**. A sampler that returns one configuration hits
  the target region 100% of the time and teaches the policy nothing that
  generalises — it will be rejected for collapsed spread.
* Parameter names must be exactly `{', '.join(SAMPLER_MODULE_REQUIRES['sample_cube_poses'])}`, in that order.

### Both files

* Import **only** `torch` and `math`, at module level. Nothing else — no
  `numpy`, no `os`, no `random`.
* No classes, no decorators, no `while` loops, no `global`.
* Under {MAX_SOURCE_LINES} lines each.
* Self-contained: the two files do not import each other.
"""


def api_surface_markdown():
    """The exact attribute surface the generated code may read."""
    env_attrs = "\n".join(f"    env.{a}" for a in sorted(ALLOWED_ENV_ATTRS))
    info_keys = "\n".join(f'    info["{k}"]' for k in sorted(ALLOWED_INFO_KEYS))
    return f"""\
These are the **only** things the generated code may read. Anything else is a
load error — the checker walks the syntax tree before the module is imported.

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
