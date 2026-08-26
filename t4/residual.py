#!/usr/bin/env python
"""The Policy Decorator residual: the head, the bound, and the agent wrapper.

Imports `torch` and nothing else - no gym, no ManiSkill - so every claim this
file makes is testable on a laptop with `t4/test_t4.py`. Same split as
t2/geometry.py against t2/harness.py: the layer where being wrong is most
expensive gets the cheapest feedback.

THE THREE THINGS THIS FILE DECIDES
----------------------------------

1. The residual is BOUNDED by alpha, as `alpha * tanh(raw)`. That is exactly
   what Policy Decorator's released code does (`res_scale * a`, where `a` is
   already tanh-squashed by SAC), and it is CLAUDE.md's `clip(delta, -a, +a)`
   with a differentiable corner. `pd_ee_delta_pos` maps a normalised +-1 to
   +-0.1 m, so alpha is a bound in METRES: alpha = 0.05 is 5 mm per step.

2. The gripper dimension is EXCLUDED FROM THE HEAD, not masked after it. The
   head emits `res_horizon * 3` numbers and there is no fourth coordinate
   anywhere - no output row, no log_std entry, no sampled dimension. The
   alternative (emit 4 and zero the last) would leave inert coordinates inside
   `log_prob.sum(-1)`, `entropy()` and `approx_kl`, so PPO would integrate noise
   over dimensions that provably cannot affect the environment: inflated
   entropy, extra variance in the importance ratio, and `target_kl` early-stops
   triggered by movement nothing depends on.

   The residual therefore cannot change the gripper COMMAND. It can still
   change gripper TIMING, because the base is closed-loop: move the end
   effector and the base's next chunk - gripper channel included - is
   conditioned on a different observation. It controls where the arm goes; the
   grasp decision follows through the base policy.

3. The base policy is never re-planned more often than it was during T-I and
   T-II. `ResidualAgent` holds the leftover of the base's 8-step chunk
   internally, so a residual that re-queries every 4 or 2 steps
   (`res_horizon < act_horizon`) still calls the frozen base exactly once per 8
   env steps. Re-planning the base more often would make it a DIFFERENT policy
   and the paired before/after would compare two policies as well as two
   heads.
"""
import math

import torch
import torch.nn as nn

RES_DIM = 3          # translation only; see (2) above
ACT_DIM = 4          # pd_ee_delta_pos: 3 translation + 1 gripper
METRES_PER_UNIT = 0.1   # panda.py:105-106, pos_lower/pos_upper on the ee controller


# ---------------------------------------------------------------------------
# the bound
# ---------------------------------------------------------------------------


def bound(raw, alpha):
    """The residual displacement, hard-bounded to (-alpha, alpha).

    Separate from the module so the bound can be tested without building one,
    and so training and evaluation cannot drift onto two versions of it.
    """
    return alpha * torch.tanh(raw)


def alpha_at(env_step, alpha, warmup):
    """Policy Decorator's progressive exploration schedule, re-derived for PPO.

    PD executes `pi_base` with probability 1-eps and `pi_base + pi_res` with
    probability eps, eps ramping 0 -> 1 over H steps (Sec 4.2). That behaviour
    policy is a MIXTURE OF A DIRAC AND A GAUSSIAN: its density is undefined, so
    PPO cannot form an importance ratio for it. Free for SAC, wrong for PPO.

    The PPO-safe analogue ramps the BOUND instead. It buys the same thing - the
    residual starts unable to deviate and is let out slowly - with log-probs
    that stay correct, because the caller holds alpha constant across a whole
    collect+update iteration so collection and update always agree.
    """
    if warmup <= 0:
        return alpha
    return alpha * min(env_step / float(warmup), 1.0)


def apply_delta(base_actions, raw, alpha):
    """base (B, T, ACT_DIM) + bounded residual on the first RES_DIM dims.

    `raw` is (B, T * RES_DIM), the head's unsquashed Gaussian sample. The
    gripper column is copied through untouched, then the whole action is
    clipped to the env's [-1, 1] box - ManiSkill would clip anyway, but doing
    it here means the action that is logged is the action that was executed.
    """
    b, t, a = base_actions.shape
    if a != ACT_DIM:
        raise ValueError(f"base actions have {a} dims, expected {ACT_DIM}")
    if raw.shape != (b, t * RES_DIM):
        raise ValueError(f"raw is {tuple(raw.shape)}, expected {(b, t * RES_DIM)}")
    out = base_actions.clone()
    out[..., :RES_DIM] = out[..., :RES_DIM] + bound(raw, alpha).view(b, t, RES_DIM)
    return out.clamp(-1.0, 1.0)


# ---------------------------------------------------------------------------
# the head
# ---------------------------------------------------------------------------


def layer_init(layer, std=math.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def _mlp(inp, hidden, out, out_std):
    return nn.Sequential(
        layer_init(nn.Linear(inp, hidden)), nn.Tanh(),
        layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
        layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
        layer_init(nn.Linear(hidden, out), std=out_std),
    )


class ResidualHead(nn.Module):
    """Actor and critic over the FROZEN base policy's observation embedding.

    Policy Decorator's structural choice, kept: the residual sees the base's
    own encoder output rather than pixels (`actor_input='obs'` in their
    released code, which they report beats concatenating the base action), so
    it is a ~400k-parameter MLP and no gradient ever reaches the vision stack
    the base policy depends on.

    `actor_mean`'s last layer is initialised at std=0.01 with zero bias and
    log_std at -1, so at iteration 0 the residual is ~0 and the behaviour IS
    the base policy. PPO never has to climb back from a random start.
    """

    def __init__(self, emb_dim, res_horizon=8, hidden=256, log_std_init=-1.0):
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.res_horizon = int(res_horizon)
        self.out_dim = self.res_horizon * RES_DIM
        self.hidden = int(hidden)
        self.actor_mean = _mlp(self.emb_dim, hidden, self.out_dim, out_std=0.01)
        self.actor_logstd = nn.Parameter(torch.ones(1, self.out_dim) * log_std_init)
        self.critic = _mlp(self.emb_dim, hidden, 1, out_std=1.0)

    def get_value(self, emb):
        return self.critic(emb)

    def _dist(self, emb):
        mean = self.actor_mean(emb)
        std = self.actor_logstd.expand_as(mean).exp()
        return torch.distributions.Normal(mean, std)

    def get_action_and_value(self, emb, action=None):
        probs = self._dist(emb)
        if action is None:
            action = probs.sample()
        return (action, probs.log_prob(action).sum(1), probs.entropy().sum(1),
                self.critic(emb))

    def act(self, emb, deterministic=True):
        """The evaluation action: the mean, no sampling.

        Deterministic on purpose. The base policy's DDPM already injects
        sampling noise into every rollout; adding the residual's own would make
        the before/after differ in how much noise it carries as well as in what
        it does.
        """
        if deterministic:
            return self.actor_mean(emb)
        return self._dist(emb).sample()


# ---------------------------------------------------------------------------
# the agent wrapper - the one path both arms go through
# ---------------------------------------------------------------------------


class ResidualAgent:
    """Frozen base + optional residual, exposing the base's own `get_action`.

    With `head=None` this is a literal passthrough, which is the point:
    t2/harness.build_agent returns one of these ALWAYS, so the before arm and
    the after arm run identical code and the comparison is two policies rather
    than two code paths (CLAUDE.md, "Reusing this harness for T-IV").

    `get_action(obs) -> (B, T, 4)` where T is `res_horizon`. Every caller in
    this repo does

        chunk = agent.get_action(obs)
        for i in range(chunk.shape[1]): envs.step(chunk[:, i])

    and sizes itself off the returned chunk, so a residual that re-queries
    every 4 steps needs no change anywhere. The base is still consulted once
    per `act_horizon` steps - the remainder of its chunk is held in `_pending`.
    """

    def __init__(self, base, head=None, alpha=0.05, act_horizon=8):
        self.base = base
        self.head = head
        self.alpha = float(alpha)
        self.act_horizon = int(act_horizon)
        self._emb_dim = None
        self._pending = None
        self._off = 0

    # -- residual management ------------------------------------------------

    def set_residual(self, head, alpha=None):
        """Swap the head between evaluation blocks.

        eval_modes.py runs three blocks under three policy seeds; T-IV pairs
        residual seed b with block b, so the head changes without the base, the
        env or the seed selection changing.
        """
        self.head = head
        if alpha is not None:
            self.alpha = float(alpha)
        self.reset_chunk()

    def reset_chunk(self):
        """Drop any unconsumed base chunk. Call at every env reset.

        Only reachable when res_horizon < act_horizon AND an episode ended
        mid-chunk; carrying a stale chunk into the next episode would apply the
        previous episode's plan to a fresh initial state.
        """
        self._pending, self._off = None, 0

    @property
    def res_horizon(self):
        return self.act_horizon if self.head is None else self.head.res_horizon

    # -- observation -> embedding ------------------------------------------

    @staticmethod
    def _permuted(obs):
        """(B, H, h, w, C) -> (B, H, C, h, w), on a shallow copy.

        Agent.get_action does this by assigning back into the dict it was
        handed. Working on a copy keeps the caller's dict clean and means the
        order of `get_action` and `embed` cannot matter.
        """
        o = dict(obs)
        for k in ("rgb", "depth"):
            if k in o:
                o[k] = o[k].permute(0, 1, 4, 2, 3)
        return o

    def embed(self, obs):
        """The base encoder's features for the MOST RECENT frame only.

        encode_obs returns (B, obs_horizon * D); Policy Decorator feeds the
        residual the last D of that. Consumes no RNG - PlainConv is conv/relu/
        pool/fc and `self.aug` is dead code in this baseline - so calling it
        alongside get_action leaves the base action bit-identical.
        """
        full = self.base.encode_obs(self._permuted(obs), eval_mode=True)
        if self._emb_dim is None:
            self._emb_dim = full.shape[1] // self.base.obs_horizon
        return full[:, -self._emb_dim:]

    def emb_dim(self, obs):
        if self._emb_dim is None:
            self.embed(obs)
        return self._emb_dim

    # -- actions ------------------------------------------------------------

    def base_chunk(self, obs):
        """The next `res_horizon` base actions, re-planning only when spent."""
        if self._pending is None:
            self._pending = self.base.get_action(dict(obs))
            self._off = 0
        t = self.res_horizon
        sl = self._pending[:, self._off:self._off + t]
        self._off += t
        if self._off >= self._pending.shape[1]:
            self._pending, self._off = None, 0
        return sl

    def get_action(self, obs):
        with torch.no_grad():
            chunk = self.base_chunk(obs)
            if self.head is None:
                return chunk
            raw = self.head.act(self.embed(obs), deterministic=True)
            return apply_delta(chunk, raw, self.alpha)

    def get_action_train(self, obs, alpha=None):
        """The collection path: base chunk, embedding, sampled residual.

        Returns (executed_actions, emb, raw, logprob, value). The base half is
        under no_grad; the head's is not.
        """
        if self.head is None:
            raise RuntimeError("get_action_train needs a residual head")
        a = self.alpha if alpha is None else float(alpha)
        with torch.no_grad():
            chunk = self.base_chunk(obs)
            emb = self.embed(obs)
            raw, logprob, _, value = self.head.get_action_and_value(emb)
            return apply_delta(chunk, raw, a), emb, raw, logprob, value.flatten()

    # -- torch plumbing -----------------------------------------------------

    def eval(self):
        self.base.eval()
        if self.head is not None:
            self.head.eval()
        return self

    def train(self, mode=True):
        self.base.eval()          # the base is frozen, always
        if self.head is not None:
            self.head.train(mode)
        return self


# ---------------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------------

# A residual is a SEPARATE small file, never merged into the base state dict.
# That is what lets t2/harness.inspect_ckpt, load_weights and verify.py check 5
# ("exactly one ckpt_sha256 per pass") keep working untouched: CKPT stays the
# frozen base and the residual arrives through its own env var.


def save_residual(path, head, alpha, act_horizon, **meta):
    torch.save(dict(head=head.state_dict(), alpha=float(alpha),
                    act_horizon=int(act_horizon),
                    res_horizon=int(head.res_horizon), res_dim=RES_DIM,
                    emb_dim=int(head.emb_dim), hidden=int(head.hidden),
                    **meta), path)


def load_residual(path, device="cpu"):
    """-> (head, alpha, act_horizon, meta). Shapes come from the file."""
    d = torch.load(path, map_location=device, weights_only=False)
    for k in ("head", "alpha", "emb_dim", "res_horizon"):
        if k not in d:
            raise ValueError(f"{path} is not a T-IV residual checkpoint "
                             f"(no '{k}'); it may be a base policy checkpoint")
    if d.get("res_dim", RES_DIM) != RES_DIM:
        raise ValueError(f"{path} was trained with res_dim={d['res_dim']}, "
                         f"this build is {RES_DIM}")
    head = ResidualHead(d["emb_dim"], res_horizon=d["res_horizon"],
                        hidden=d.get("hidden", 256)).to(device)
    head.load_state_dict(d["head"])
    head.eval()
    meta = {k: v for k, v in d.items() if k != "head"}
    return head, float(d["alpha"]), int(d.get("act_horizon", 8)), meta
