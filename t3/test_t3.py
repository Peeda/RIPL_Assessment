#!/usr/bin/env python
"""Self-tests for the contract and the checker. No deps, ~1 s.

    python3 t3/test_t3.py

A checker that has only ever been shown good input is a checker nobody has
tried to fool. This asserts three things, and the third is the one that carries
the design decision:

    1. the three committed fixtures LOAD - the positive control;
    2. each broken snippet is an ERROR, naming the right reason;
    3. each merely-questionable snippet is a WARNING and still loads.

(3) is the whole "allow most responses to proceed" policy, expressed as a test.
If a rule migrates from warning to error, this file fails and someone has to
decide that on purpose.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from loader import check_static, load_source  # noqa: E402
from spec import auc, auc_z, se_null_auc  # noqa: E402

PASS = FAIL = 0


def check(ok, label, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL  {label}   {detail}")


REWARD_HEAD = "import torch\n\nREWARD_MAX = 8.0\n\n"
GOOD_BODY = ("def compute_reward(env, obs, action, info):\n"
             "    return torch.zeros(env.num_envs)\n")
GOOD_SAMPLER = (
    "import torch\n\n"
    "def sample_cube_poses(b, device):\n"
    "    p = torch.zeros((b, 3))\n"
    "    q = torch.zeros((b, 4))\n"
    "    return {'cubeA_xyz': p, 'cubeA_quat': q,\n"
    "            'cubeB_xyz': p, 'cubeB_quat': q}\n")

# (label, kind, source, substring the error must name)
ERRORS = [
    ("syntax", "reward", REWARD_HEAD + "def compute_reward(env, obs,\n", "parse"),
    ("import os", "reward", "import os\n" + REWARD_HEAD + GOOD_BODY,
     "forbidden import"),
    # The rule that matters most: numpy draws from a stream reset(seed=s) does
    # not seed, so T-IV's training distribution stops being reproducible.
    ("sampler imports numpy", "sampler",
     "import numpy as np\n" + GOOD_SAMPLER, "forbidden import"),
    ("arg order", "reward",
     REWARD_HEAD + "def compute_reward(env, action, obs, info):\n"
     "    return torch.zeros(env.num_envs)\n", "names and order"),
    ("missing function", "reward", REWARD_HEAD + "def other(a):\n    return a\n",
     "missing required function"),
    ("missing REWARD_MAX", "reward", "import torch\n\n" + GOOD_BODY,
     "REWARD_MAX"),
    ("REWARD_MAX is an expression", "reward",
     "import torch\n\nREWARD_MAX = 4.0 * 2\n\n" + GOOD_BODY, "numeric literal"),
    ("while loop", "sampler",
     "import torch\n\ndef sample_cube_poses(b, device):\n"
     "    while True:\n        pass\n", "while"),
    ("mutates the sim", "reward",
     REWARD_HEAD + "def compute_reward(env, obs, action, info):\n"
     "    env.cubeA.set_pose(env.cubeB.pose)\n"
     "    return torch.zeros(env.num_envs)\n", "mutates"),
    ("exec", "reward",
     REWARD_HEAD + "def compute_reward(env, obs, action, info):\n"
     "    exec('x = 1')\n    return torch.zeros(env.num_envs)\n", "exec"),
    ("class", "reward",
     REWARD_HEAD + "class R:\n    pass\n\n" + GOOD_BODY, "ClassDef"),
    ("dunder escape", "reward",
     REWARD_HEAD + "def compute_reward(env, obs, action, info):\n"
     "    return env.__class__\n", "dunder"),
]

# (label, kind, source, substring the WARNING must name). Each must ALSO load:
# these are things to notice, not things to refuse.
WARNINGS = [
    ("gpu sync", "reward",
     REWARD_HEAD + "def compute_reward(env, obs, action, info):\n"
     "    x = env.num_envs\n"
     "    return torch.zeros(x) + float(torch.zeros(1).cpu()[0])\n", "sync"),
    ("off-surface attribute", "reward",
     REWARD_HEAD + "def compute_reward(env, obs, action, info):\n"
     "    return torch.zeros(env.num_envs) + env.elapsed_steps\n",
     "outside the documented API surface"),
    ("unknown info key", "reward",
     REWARD_HEAD + "def compute_reward(env, obs, action, info):\n"
     "    return torch.zeros(env.num_envs) + info['is_grasped']\n",
     "not a key evaluate() returns"),
]

FIXTURES = [("good_reward.py", "reward"), ("good_sampler.py", "sampler"),
            ("stock_reward.py", "reward")]

# The exec half of the loader needs torch, which this laptop does not have.
# check_static is the part that must run on the bare interpreter, so the import
# is optional and its absence is reported rather than failing the suite.
try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


def test_fixtures_load():
    for name, kind in FIXTURES:
        path = os.path.join(HERE, "fixtures", name)
        with open(path) as f:
            src = f.read()
        errors, _ = check_static(src, kind, name)
        check(not errors, f"fixture {name} is clean", str(errors[:1]))
        if not HAVE_TORCH:
            continue
        try:
            load_source(src, kind, name)
            check(True, f"fixture {name} loads")
        except Exception as e:                                    # noqa: BLE001
            check(False, f"fixture {name} loads", f"{type(e).__name__}: {e}")


def test_errors():
    for label, kind, src, want in ERRORS:
        errors, _ = check_static(src, kind, label)
        if not errors:
            check(False, f"error: {label}", "ACCEPTED - the checker has a hole")
            continue
        check(any(want in e for e in errors), f"error: {label}",
              f"rejected, but for the wrong reason: {errors}")


def test_warnings_still_load():
    for label, kind, src, want in WARNINGS:
        errors, warnings = check_static(src, kind, label)
        check(not errors, f"warning: {label} is not an error", str(errors[:1]))
        check(any(want in w for w in warnings), f"warning: {label} is flagged",
              f"warnings were {warnings}")


def test_auc():
    # perfect separation, then perfectly inverted, then chance
    check(auc([1, 2, 3, 4], [0, 0, 1, 1]) == 1.0, "auc perfect")
    check(auc([4, 3, 2, 1], [0, 0, 1, 1]) == 0.0, "auc inverted")
    check(auc([1, 2, 3, 4], [0, 1, 1, 0]) == 0.5, "auc chance")
    # all tied is 0.5 by the half-credit convention
    check(auc([2, 2, 2, 2], [0, 0, 1, 1]) == 0.5, "auc all ties")
    # one tie straddling the boundary: pairs are (3>1),(3>2),(2==2 -> .5),(2>1)
    check(abs(auc([1, 2, 2, 3], [0, 0, 1, 1]) - 0.875) < 1e-12, "auc one tie",
          str(auc([1, 2, 2, 3], [0, 0, 1, 1])))
    # a statistic with no data returns nan, never a number that reads as a
    # measurement - the geometry.wilson(0, 0) precedent
    a = auc([1, 2, 3], [1, 1, 1])
    check(a != a, "auc with one empty class is nan")
    s = se_null_auc(50, 50)
    check(abs(s - 0.0580) < 1e-3, "se_null_auc at n=50/50", str(s))
    check(se_null_auc(0, 10) != se_null_auc(0, 10), "se_null_auc(0, n) is nan")
    check(abs(auc_z(1.0, 50, 50) - 0.5 / s) < 1e-9, "auc_z")


def main():
    for t in (test_fixtures_load, test_errors, test_warnings_still_load,
              test_auc):
        t()
    if not HAVE_TORCH:
        print("\n  (no torch here - check_static was tested, the "
              "guarded exec was not)")
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
