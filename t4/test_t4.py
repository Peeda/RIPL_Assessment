#!/usr/bin/env python
"""t4/residual.py against hand-computed cases. torch + stdlib, ~2 s, no sim.

The layer this covers - the bound, the gripper exclusion, the passthrough, the
base's re-planning rate - is the layer where being wrong is both most expensive
and least visible: every one of these would produce a plausible CSV.

    python3 t4/test_t4.py
"""
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from residual import (ACT_DIM, RES_DIM, ResidualAgent, ResidualHead,  # noqa: E402
                      alpha_at, apply_delta, bound, load_residual,
                      save_residual)

N = 0


def ok(cond, msg):
    global N
    N += 1
    if not cond:
        print(f"FAIL  {msg}")
        sys.exit(1)


def close(a, b, tol=1e-6):
    return torch.allclose(torch.as_tensor(a), torch.as_tensor(b), atol=tol)


# ---------------------------------------------------------------------------
# a stand-in for train_rgbd.Agent: the same surface, none of the weight
# ---------------------------------------------------------------------------

D_VIS, D_STATE, OBS_H, ACT_H = 256, 29, 2, 8


class FakeBase:
    def __init__(self, act_horizon=ACT_H):
        self.obs_horizon = OBS_H
        self.act_horizon = act_horizon
        self.calls = 0

    def encode_obs(self, obs_seq, eval_mode):
        b = obs_seq["state"].shape[0]
        # deterministic, and different per frame so "most recent" is checkable
        f = torch.arange(OBS_H, dtype=torch.float32).view(1, OBS_H, 1)
        base = obs_seq["state"].sum(-1, keepdim=True) + f
        return base.expand(b, OBS_H, D_VIS + D_STATE).flatten(1).contiguous()

    def get_action(self, obs_seq):
        self.calls += 1
        # mutates the dict it is handed, exactly as the real one does
        obs_seq["rgb"] = obs_seq["rgb"].permute(0, 1, 4, 2, 3)
        b = obs_seq["state"].shape[0]
        g = torch.Generator().manual_seed(self.calls)
        return torch.rand((b, self.act_horizon, ACT_DIM), generator=g) * 1.6 - 0.8

    def eval(self):
        return self


def fake_obs(b=3):
    return dict(state=torch.randn(b, OBS_H, D_STATE),
                rgb=torch.randint(0, 255, (b, OBS_H, 16, 16, 6), dtype=torch.uint8))


# ---------------------------------------------------------------------------
# 1. the bound
# ---------------------------------------------------------------------------

a = 0.05
raw = torch.linspace(-50, 50, 401).view(1, -1)
d = bound(raw, a)
ok(bool((d.abs() <= a).all()), "bound must never exceed alpha")
# tanh SATURATES: at |raw| ~ 20 in float32 it returns exactly 1.0, so the bound
# is ATTAINED, not approached - and the attained value is float32(alpha), which
# is one ULP ABOVE the python double 0.05. Compare in float32 or this reads as
# a 7.5e-10 overshoot. Do not "fix" it by loosening the <= above.
ok(float(d.abs().max()) == float(torch.tensor(a, dtype=torch.float32)),
   "the bound is attained at saturation, never passed")
mid = bound(torch.linspace(-5, 5, 101).view(1, -1), a)
ok(bool((mid.abs() < a).all()), "away from saturation the bound is strict")
ok(close(bound(torch.zeros(1, 4), a), torch.zeros(1, 4)), "zero raw -> zero delta")
ok(bool((d.diff() >= 0).all()), "bound is non-decreasing in raw")
ok(bool((mid.diff() > 0).all()), "and strictly increasing away from saturation")
ok(close(bound(raw, 0.0), torch.zeros_like(raw)), "alpha=0 -> exactly no residual")
# alpha is a bound in metres via pd_ee_delta_pos's +-0.1 m mapping
ok(abs(a * 0.1 - 0.005) < 1e-12, "alpha=0.05 is 5 mm per step")

# ---------------------------------------------------------------------------
# 2. the alpha ramp (PD's progressive exploration, PPO-safe form)
# ---------------------------------------------------------------------------

ok(alpha_at(0, 0.05, 1000) == 0.0, "ramp starts at zero")
ok(abs(alpha_at(500, 0.05, 1000) - 0.025) < 1e-12, "ramp is linear")
ok(alpha_at(1000, 0.05, 1000) == 0.05, "ramp reaches alpha at H")
ok(alpha_at(10_000, 0.05, 1000) == 0.05, "ramp clamps above H")
ok(alpha_at(0, 0.05, 0) == 0.05, "warmup<=0 disables the ramp")
prev = -1.0
for s in range(0, 2000, 37):
    v = alpha_at(s, 0.05, 1000)
    ok(v >= prev, "ramp is non-decreasing")
    prev = v

# ---------------------------------------------------------------------------
# 3. apply_delta: the gripper column, and the box
# ---------------------------------------------------------------------------

torch.manual_seed(0)
base = torch.rand(4, ACT_H, ACT_DIM) * 1.6 - 0.8
r = torch.randn(4, ACT_H * RES_DIM) * 10
out = apply_delta(base, r, a)
ok(out.shape == base.shape, "apply_delta preserves shape")
ok(close(out[..., 3], base[..., 3]), "the GRIPPER column is untouched")
ok(bool(((out[..., :3] - base[..., :3]).abs() <= a + 1e-6).all()),
   "every translation dim moves by at most alpha")
ok(bool((out.abs() <= 1.0 + 1e-6).all()), "the action stays inside the [-1,1] box")
ok(close(apply_delta(base, r, 0.0), base), "alpha=0 is the identity")
# clipping must not smuggle the gripper out of range either
big = torch.full((2, ACT_H, ACT_DIM), 3.0)
ok(bool((apply_delta(big, torch.zeros(2, ACT_H * RES_DIM), a) == 1.0).all()),
   "out-of-box base actions are clipped, gripper included")

try:
    apply_delta(base, torch.randn(4, ACT_H * ACT_DIM), a)
    ok(False, "a 4-dim raw must be rejected")
except ValueError:
    ok(True, "a 4-dim raw is rejected")

# ---------------------------------------------------------------------------
# 4. the head has NO fourth coordinate anywhere
# ---------------------------------------------------------------------------

emb_dim = D_VIS + D_STATE
head = ResidualHead(emb_dim, res_horizon=ACT_H)
ok(head.out_dim == ACT_H * RES_DIM, "head emits res_horizon x 3")
ok(head.actor_mean[-1].out_features == ACT_H * RES_DIM,
   "the final Linear has no gripper output row")
ok(tuple(head.actor_logstd.shape) == (1, ACT_H * RES_DIM),
   "log_std has no gripper entry")
emb = torch.randn(5, emb_dim)
_, lp, ent, _ = head.get_action_and_value(emb)
ok(lp.shape == (5,) and ent.shape == (5,), "log_prob and entropy are per-sample")
ok(head.get_value(emb).shape == (5, 1), "the critic is a scalar per sample")

# 5. near-zero init: iteration 0 IS the base policy
mean = head.act(emb, deterministic=True)
ok(float(bound(mean.detach(), a).abs().max()) < 1e-3,
   "at init the deterministic residual is ~0, so training starts at the base")

# ---------------------------------------------------------------------------
# 6. passthrough: head=None must be bit-identical to the bare base
# ---------------------------------------------------------------------------

o = fake_obs()
b1, b2 = FakeBase(), FakeBase()
want = b1.get_action(dict(o))
got = ResidualAgent(b2, head=None).get_action(o)
ok(close(want, got, 0.0), "head=None is a BIT-IDENTICAL passthrough")
ok("rgb" in o and o["rgb"].shape[-1] == 6,
   "the wrapper does not mutate the caller's observation dict")

# alpha=0 with a real head is also the identity
b3 = FakeBase()
ag0 = ResidualAgent(b3, head=ResidualHead(emb_dim, res_horizon=ACT_H), alpha=0.0)
ok(close(ag0.get_action(o), FakeBase().get_action(dict(o)), 0.0),
   "alpha=0 with a head attached is still the base policy")

# ---------------------------------------------------------------------------
# 7. the base is re-planned once per act_horizon, whatever res_horizon is
# ---------------------------------------------------------------------------

for rh in (8, 4, 2, 1):
    fb = FakeBase()
    ag = ResidualAgent(fb, ResidualHead(emb_dim, res_horizon=rh), alpha=a)
    steps = 0
    for _ in range(ACT_H * 3 // rh):
        c = ag.get_action(fake_obs())
        ok(c.shape[1] == rh, f"res_horizon={rh}: chunk is {rh} steps")
        steps += rh
    ok(steps == ACT_H * 3, f"res_horizon={rh}: 24 env steps produced")
    ok(fb.calls == 3, f"res_horizon={rh}: the BASE re-planned 3 times, not {fb.calls}")

# the slices really are consecutive pieces of one base chunk
fb = FakeBase()
ag = ResidualAgent(fb, ResidualHead(emb_dim, res_horizon=4), alpha=0.0)
first, second = ag.get_action(fake_obs()), ag.get_action(fake_obs())
whole = FakeBase().get_action(dict(fake_obs()))
ok(close(torch.cat([first, second], 1), whole, 0.0),
   "consecutive residual steps consume one base chunk in order")

# reset_chunk drops a half-consumed plan
fb = FakeBase()
ag = ResidualAgent(fb, ResidualHead(emb_dim, res_horizon=4), alpha=a)
ag.get_action(fake_obs())
ag.reset_chunk()
ag.get_action(fake_obs())
ok(fb.calls == 2, "reset_chunk forces a fresh base plan")

# ---------------------------------------------------------------------------
# 8. the embedding is the MOST RECENT frame, at the right width
# ---------------------------------------------------------------------------

ag = ResidualAgent(FakeBase(), None)
o = fake_obs(2)
e = ag.embed(o)
ok(e.shape == (2, emb_dim), "embed returns one frame's width")
full = FakeBase().encode_obs(ResidualAgent._permuted(o), True)
ok(close(e, full[:, -emb_dim:]), "embed takes the LAST frame, not the first")

# ---------------------------------------------------------------------------
# 9. checkpoints
# ---------------------------------------------------------------------------

with tempfile.TemporaryDirectory() as td:
    p = os.path.join(td, "r.pt")
    h = ResidualHead(emb_dim, res_horizon=4, hidden=128)
    with torch.no_grad():
        for q in h.parameters():
            q.add_(torch.randn_like(q) * 0.1)
    save_residual(p, h, alpha=0.03, act_horizon=8, mode="gap", seed=2,
                  base_sha256="deadbeef")
    h2, al, ah, meta = load_residual(p)
    ok(al == 0.03 and ah == 8, "alpha and act_horizon round-trip")
    ok(meta["mode"] == "gap" and meta["seed"] == 2, "metadata round-trips")
    ok(meta["res_dim"] == RES_DIM, "res_dim is recorded")
    ok(h2.res_horizon == 4 and h2.hidden == 128, "shape comes from the file")
    x = torch.randn(3, emb_dim)
    ok(close(h.act(x), h2.act(x)), "the reloaded head acts identically")

    bad = os.path.join(td, "base.pt")
    torch.save({"ema_agent": {"visual_encoder.fc.0.weight": torch.zeros(256, 8192)}}, bad)
    try:
        load_residual(bad)
        ok(False, "a base checkpoint must be rejected")
    except ValueError as e:
        ok("residual checkpoint" in str(e), "a base checkpoint is rejected by name")

# ---------------------------------------------------------------------------
# 10. gradients reach the head and stop at the base
# ---------------------------------------------------------------------------

h = ResidualHead(emb_dim, res_horizon=ACT_H)
emb = torch.randn(6, emb_dim)
act, lp, ent, val = h.get_action_and_value(emb)
(lp.sum() + val.sum() + ent.sum()).backward()
ok(all(q.grad is not None for q in h.parameters()), "every head parameter has a gradient")
ok(h.actor_logstd.grad is not None, "log_std is learned")

fb = FakeBase()
ag = ResidualAgent(fb, ResidualHead(emb_dim, res_horizon=ACT_H), alpha=a)
exe, e2, r2, lp2, v2 = ag.get_action_train(fake_obs())
ok(exe.shape == (3, ACT_H, ACT_DIM), "the training path executes a full chunk")
ok(not exe.requires_grad, "collection is under no_grad; PPO recomputes log-probs")
ok(close(exe[..., 3], FakeBase().get_action(dict(fake_obs()))[..., 3], 0.0),
   "gripper untouched on the training path too")
ok(e2.shape == (3, emb_dim) and r2.shape == (3, ACT_H * RES_DIM),
   "the training path returns the embedding and the raw residual")
ok(lp2.shape == (3,) and v2.shape == (3,), "log-prob and value are per-env")


# ---------------------------------------------------------------------------
# 11. the t2/harness.py seam, without a simulator
#
# harness.py's only unconditional non-stdlib imports are gymnasium, numpy,
# torch and residual, and CubePoseInfo just needs `gym.Wrapper` to exist as a
# base class. Stubbing gymnasium is therefore enough to import the module and
# exercise the parts T-IV added - which is worth doing off-pod, because an
# ImportError or a bad $RESIDUAL path otherwise costs a whole pod cycle.
# ---------------------------------------------------------------------------

import types  # noqa: E402

_gym = types.ModuleType("gymnasium")
_gym.Wrapper = type("Wrapper", (), {"__init__": lambda self, env: None})
sys.modules.setdefault("gymnasium", _gym)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "t2"))
import harness  # noqa: E402

os.environ.pop("RESIDUAL", None)
ok(harness.residual_path() is None, "no RESIDUAL -> no residual")
ok(harness.residual_path(1) is None, "no RESIDUAL -> none per block either")

with tempfile.TemporaryDirectory() as td:
    paths = {}
    for b in (1, 2, 3):
        q = os.path.join(td, f"residual_seed{b}.pt")
        h = ResidualHead(emb_dim, res_horizon=ACT_H)
        with torch.no_grad():
            h.actor_mean[-1].weight.add_(float(b))    # a distinguishable head
        save_residual(q, h, alpha=0.05, act_horizon=ACT_H, mode="gap", seed=b)
        paths[b] = q

    os.environ["RESIDUAL"] = paths[2]
    ok(harness.residual_path() == paths[2], "a plain RESIDUAL resolves eagerly")
    ok(harness.residual_path(1) == paths[2], "and ignores the block")

    os.environ["RESIDUAL"] = os.path.join(td, "residual_seed{block}.pt")
    ok(harness.residual_path() is None,
       "a {block} template does NOT resolve without a block")
    for b in (1, 2, 3):
        ok(harness.residual_path(b) == paths[b], f"{{block}} -> block {b}")

    # sha16 is the same function manifest() hashes the checkpoint with, so a
    # residual number is attributable the same way a base number is
    ok(len(harness.sha16(paths[1])) == 16, "sha16 returns 16 hex chars")
    ok(harness.sha16(paths[1]) != harness.sha16(paths[2]),
       "different heads hash differently")
    ok(harness.sha16(os.path.join(td, "nope.pt")) == "unknown",
       "a missing file hashes to 'unknown', it does not raise")

    # attach_residual swaps the head and refuses an act_horizon mismatch
    ag = ResidualAgent(FakeBase(), head=None, act_horizon=ACT_H)
    m = harness.attach_residual(ag, paths[3], "cpu")
    ok(ag.head is not None and m["seed"] == 3, "attach_residual loads the head")
    e = torch.randn(2, emb_dim)
    for b in (1, 2, 3):
        harness.attach_residual(ag, paths[b], "cpu")
        ok(close(ag.head.act(e), load_residual(paths[b])[0].act(e)),
           f"block {b} really gets block {b}'s head")

    bad = ResidualAgent(FakeBase(act_horizon=4), head=None, act_horizon=4)
    try:
        harness.attach_residual(bad, paths[1], "cpu")
        ok(False, "an act_horizon mismatch must be refused")
    except SystemExit:
        ok(True, "an act_horizon mismatch is refused, not silently accepted")

os.environ.pop("RESIDUAL", None)
os.environ["RESIDUAL"] = "/nonexistent/residual.pt"
try:
    harness.residual_path()
    ok(False, "a missing RESIDUAL must be refused")
except SystemExit:
    ok(True, "a missing RESIDUAL is refused before any rollout runs")
os.environ.pop("RESIDUAL", None)

print(f"\n{N} assertions passed.")
