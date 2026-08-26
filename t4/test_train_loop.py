#!/usr/bin/env python
"""Run t4/train_ppo.py's whole loop against a FAKE simulator.

Everything else in t4/ is testable because it is torch-only. train_ppo.py is
not: it needs ManiSkill, a GPU env and a 33 MB diffusion policy. So it is
exactly the file where a shape error, an unbound name or an off-by-one hides
until a pod is running - and a pod cycle costs more than this file does.

So: stub gymnasium and mani_skill, stand in a fake base policy with the same
surface train_rgbd.Agent presents (encode_obs / get_action / obs_horizon), hand
it a fake vector env with StackCube's real shapes, and run main() end to end for
a few iterations. What that proves is NOT that PPO learns - the fake env has no
dynamics - but that every tensor lines up, GAE runs, the update runs, the
checkpoint round-trips through t2/harness's loader, and the CSV is written.

    nix-shell -p "python3.withPackages(ps: [ps.torch ps.numpy ps.tyro])" \\
      --run "python3 t4/test_train_loop.py"
"""
import csv
import dataclasses
import json
import os
import sys
import tempfile
import types

try:
    import torch
    import tyro  # noqa: F401
except ModuleNotFoundError as e:
    sys.exit(f"\nt4/test_train_loop.py needs torch and tyro ({e}). On the laptop:\n"
             '    nix-shell -p "python3.withPackages(ps: [ps.torch ps.numpy ps.tyro])" \\\n'
             '      --run "python3 t4/test_train_loop.py"\n')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

N = 0


def ok(cond, msg):
    global N
    N += 1
    if not cond:
        print(f"FAIL  {msg}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# the stubs
# ---------------------------------------------------------------------------

OBS_H, ACT_H, ACT_DIM, D_VIS, D_STATE, HORIZON = 2, 8, 4, 256, 29, 200


class Box:
    def __init__(self, shape):
        self.shape = shape
        self.low = -torch.ones(shape).numpy()
        self.high = torch.ones(shape).numpy()


_gym = types.ModuleType("gymnasium")
_gym.Wrapper = type("Wrapper", (), {"__init__": lambda self, env: None})
_gym.ObservationWrapper = _gym.Wrapper
_gym.__version__ = "0.29.1"
_gym.make = lambda *a, **k: None
sys.modules.setdefault("gymnasium", _gym)

for m in ("mani_skill", "mani_skill.utils", "mani_skill.utils.wrappers",
          "mani_skill.utils.wrappers.flatten", "mani_skill.vector",
          "mani_skill.vector.wrappers", "mani_skill.vector.wrappers.gymnasium",
          "mani_skill.envs", "mani_skill.envs.tasks",
          "mani_skill.envs.tasks.tabletop",
          "mani_skill.envs.tasks.tabletop.stack_cube",
          "mani_skill.utils.registration", "mani_skill.utils.structs",
          "mani_skill.utils.structs.pose"):
    sys.modules.setdefault(m, types.ModuleType(m))
sys.modules["mani_skill.utils.wrappers"].FrameStack = object
sys.modules["mani_skill.utils.wrappers.flatten"].FlattenRGBDObservationWrapper = object
sys.modules["mani_skill.vector.wrappers.gymnasium"].ManiSkillVectorEnv = object
sys.modules["mani_skill.envs.tasks.tabletop.stack_cube"].StackCubeEnv = object
sys.modules["mani_skill.utils.registration"].register_env = lambda *a, **k: (lambda c: c)
sys.modules["mani_skill.utils.structs.pose"].Pose = object


class FakeEnvs:
    """StackCube's real shapes; no dynamics. Truncates every HORIZON steps,
    all envs together, exactly as ignore_terminations=True does."""

    def __init__(self, n):
        self.num_envs = n
        self.single_observation_space = {"state": Box((OBS_H, D_STATE)),
                                         "rgb": Box((OBS_H, 128, 128, 6))}
        self.single_action_space = Box((ACT_DIM,))
        self.t = 0
        self.closed = False
        self.actions_seen = []

    def _obs(self):
        return {"state": torch.randn(self.num_envs, OBS_H, D_STATE),
                "rgb": torch.randint(0, 255, (self.num_envs, OBS_H, 128, 128, 6),
                                     dtype=torch.uint8)}

    def reset(self, seed=None, options=None):
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        ok(tuple(action.shape) == (self.num_envs, ACT_DIM),
           f"env received a per-step action, got {tuple(action.shape)}")
        ok(bool((action.abs() <= 1.0 + 1e-6).all()),
           "every executed action is inside the [-1, 1] box")
        self.actions_seen.append(action.detach().clone())
        self.t += 1
        n = self.num_envs
        obs = self._obs()
        rew = torch.rand(n)                      # normalized_dense is in [0,1]
        term = torch.zeros(n, dtype=torch.bool)  # ignore_terminations
        trunc = torch.full((n,), self.t % HORIZON == 0)
        infos = {}
        if bool(trunc.any()):
            infos["_final_info"] = trunc.clone()
            infos["final_info"] = {"episode": {
                "success_once": torch.rand(n), "success_at_end": torch.rand(n),
                "return": torch.rand(n) * 100}}
            infos["final_observation"] = obs
            obs = self._obs()
        return obs, rew, term, trunc, infos

    def close(self):
        self.closed = True


class FakeAgent(torch.nn.Module):
    """The surface train_rgbd.Agent presents to t4/residual.py."""

    def __init__(self, envs, args):
        super().__init__()
        self.obs_horizon = args.obs_horizon
        self.act_horizon = args.act_horizon
        self.enc = torch.nn.Linear(D_STATE, D_VIS + D_STATE)
        # inspect_ckpt probes visual_encoder.fc.0.weight to tell the pooled
        # encoder from the spatial one, and load_weights is STRICT - so the
        # stub has to carry that key for real, not just in the file.
        self.visual_encoder = torch.nn.Module()
        self.visual_encoder.fc = torch.nn.Sequential(torch.nn.Linear(8192, 256))
        self.calls = 0

    def encode_obs(self, obs_seq, eval_mode):
        ok(obs_seq["rgb"].shape[2] == 6,
           "encode_obs sees rgb already permuted to (B, H, C, h, w)")
        return self.enc(obs_seq["state"].float()).flatten(1)

    def get_action(self, obs_seq):
        self.calls += 1
        b = obs_seq["state"].shape[0]
        return torch.rand((b, self.act_horizon, ACT_DIM)) * 1.6 - 0.8


@dataclasses.dataclass
class FakeArgs:
    env_id: str = ""
    demo_path: str = ""
    control_mode: str = ""
    sim_backend: str = ""
    max_episode_steps: int = HORIZON
    obs_mode: str = "rgb"
    pool_feature_map: bool = False
    obs_horizon: int = OBS_H
    act_horizon: int = ACT_H
    pred_horizon: int = 16


_tr = types.ModuleType("train_rgbd")
_tr.Args = FakeArgs
_tr.Agent = FakeAgent
sys.modules["train_rgbd"] = _tr

_e4 = types.ModuleType("env_t4")
_e4.describe = lambda: "env StackCube-T4-v1  (stubbed)"
_e4.nominal_frac = lambda: 0.0
sys.modules["env_t4"] = _e4

MADE = {}


def _make(env_id, num_envs, **kw):
    MADE.update(env_id=env_id, num_envs=num_envs, **kw)
    return FakeEnvs(num_envs)


_me = types.ModuleType("make_envs")
_me.make_train_envs = _make
sys.modules["make_envs"] = _me

# ---------------------------------------------------------------------------

os.environ["MANISKILL_REPO"] = "/nonexistent"
os.environ["T3_RUN"] = "/nonexistent"
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "t2"))
import residual as R  # noqa: E402
import train_ppo  # noqa: E402

with tempfile.TemporaryDirectory() as td:
    ckpt = os.path.join(td, "base.pt")
    _sd = FakeAgent(None, FakeArgs()).state_dict()
    ok("visual_encoder.fc.0.weight" in _sd,
       "the stub carries the key inspect_ckpt probes")
    ok(_sd["visual_encoder.fc.0.weight"].shape[1] == 8192,
       "and it is the SPATIAL shape, so the patched-encoder path is exercised")
    torch.save({"ema_agent": _sd}, ckpt)
    out = os.path.join(td, "run")

    NE, ITERS = 4, 3
    env_per_iter = NE * (HORIZON // ACT_H) * ACT_H
    sys.argv = ["train_ppo.py", "--mode", "gap", "--seed", "2", "--ckpt", ckpt,
                "--out", out, "--num-envs", str(NE),
                "--total-timesteps", str(env_per_iter * ITERS),
                "--alpha", "0.05", "--alpha-warmup", str(env_per_iter),
                "--num-minibatches", "2", "--update-epochs", "2",
                "--save-freq", "1", "--no-cuda"]
    train_ppo.main()

    # -- what the run should have produced ------------------------------
    ok(MADE["env_id"] == "StackCube-T4-v1", "the T-IV env id was used")
    ok(MADE["reward_mode"] == "normalized_dense",
       "the generated reward is used normalised, so PPO needs no reward scaling")
    # train_ppo relies on make_train_envs' own default here rather than
    # passing it, so assert the default rather than the call.
    import inspect as _i
    import make_envs as _real
    _real_mod = _i.getmodule(_make)
    _src = open(os.path.join(HERE, "make_envs.py")).read()
    ok("reconfiguration_freq=None" in _src,
       "make_train_envs defaults reconfiguration_freq to None - the scene is "
       "not rebuilt on every reset during training")
    ok("ignore_terminations=True" in _src,
       "and ignore_terminations defaults on, so episodes run the full horizon "
       "and 200/8 = 25 chunks exactly")
    ok(MADE["max_episode_steps"] == HORIZON, "the 200-step horizon is kept")

    rp = os.path.join(out, "residual_seed2.pt")
    ok(os.path.exists(rp), "a residual checkpoint was written")
    head, alpha, ah, meta = R.load_residual(rp)
    ok(alpha == 0.05 and ah == ACT_H, "alpha and act_horizon round-trip")
    ok(meta["mode"] == "gap" and meta["seed"] == 2, "mode and seed are recorded")
    ok(meta["res_horizon"] == ACT_H, "res_horizon defaulted to act_horizon")
    ok(head.out_dim == ACT_H * R.RES_DIM,
       "the saved head has no gripper dimension")

    lp = os.path.join(out, "gap_seed2_train.csv")
    ok(os.path.exists(lp), "a training CSV was written")
    rows = list(csv.DictReader(open(lp)))
    ok(len(rows) == ITERS, f"one row per iteration ({len(rows)} != {ITERS})")
    for k in ("charts/alpha_mm", "charts/delta_norm_mm", "charts/reward_mean",
              "losses/policy_loss", "losses/value_loss", "losses/approx_kl",
              "losses/entropy", "sys/vram_max_gb", "sys/wall_seconds",
              "charts/SPS", "train/success_once"):
        ok(k in rows[0], f"the log carries '{k}'")
    steps = [int(r["global_step"]) for r in rows]
    ok(steps == sorted(steps) and steps[0] == env_per_iter,
       "global_step counts ENV steps and advances one iteration at a time")
    a_mm = [float(r["charts/alpha_mm"]) for r in rows]
    ok(a_mm[0] == 0.0, "the alpha ramp starts the first iteration at zero")
    ok(a_mm == sorted(a_mm) and abs(a_mm[-1] - 5.0) < 1e-6,
       "the ramp is monotone and reaches 5 mm")
    rw = [float(r["charts/reward_mean"]) for r in rows]
    ok(all(0.0 <= x <= 1.0 for x in rw),
       "the chunk reward is the MEAN of the sub-step rewards, so it stays in "
       "[0,1] - a sum would come out ~8x larger and silently rescale every "
       "advantage")
    d_mm = [float(r["charts/delta_norm_mm"]) for r in rows]
    ok(d_mm[0] == 0.0, "with alpha=0 the applied delta is exactly zero")
    ok(all(d <= a + 1e-6 for d, a in zip(d_mm, a_mm)),
       "the applied delta never exceeds the bound in force at the time")

    mp = os.path.join(out, "gap_seed2_manifest.json")
    m = json.load(open(mp))
    ok(m["env_steps"] == env_per_iter * ITERS, "every planned env step was taken")
    ok(m["t3_sampler"] == 1, "the manifest records that the biased sampler was on")
    ok(m["t4_nominal_frac"] == 0.0, "and that no nominal episodes were mixed in")
    ok("wall_seconds" in m and "vram_max_gb" in m,
       "wall-clock and VRAM are recorded - both are named deliverables")
    ok(m["act_horizon"] == ACT_H and m["res_horizon"] == ACT_H,
       "the horizons are recorded")

print(f"\n{N} assertions passed.")
