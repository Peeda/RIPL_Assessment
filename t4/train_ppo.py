#!/usr/bin/env python
"""PPO for the Policy Decorator residual, on T-III's reward and sampler.

Adapted from $MANISKILL_REPO/examples/baselines/ppo/ppo.py. GAE, the clipped
surrogate, advantage normalisation, value clipping, the KL early-stop and the
logging are upstream's; what changes is what the MDP is.

THE MDP IS CHUNKED - one PPO timestep is one base-policy call
-------------------------------------------------------------

The base predicts pred_horizon=16, EMITS act_horizon=8, and every rollout loop
in this repo executes all 8 open-loop before calling it again. PPO is per-step,
so the two must be reconciled. We lift the MDP to the chunk:

    state       the frozen base's embedding at the replan boundary
    action      res_horizon*3 raw Gaussian; delta = alpha*tanh(.) on translation
    reward      MEAN of the per-step normalized_dense rewards, in [0, 1]
    episode     200 / 8 = exactly 25 chunked steps, no ragged remainder
    gamma       PER CHUNK. 0.9 here is 0.9**(1/8) ~ 0.987 per env step.

Why not per-step. (1) A per-step MDP is not even well defined without changing
the base policy: one action per observation means re-running the DDPM every
step and keeping only its first action, which re-plans 8x more often and is a
DIFFERENT policy - T-I's 0.730 and T-II's 0.513/0.623 would stop describing the
"before" arm and the paired comparison would collapse. (2) ManiSkill's DP has
no DDIM path (num_diffusion_iters=100), so chunking amortises 100 U-Net
forwards over 8 env steps; per-step is 100 per step. (3) 25 decisions per
episode instead of 200 shortens the span GAE must carry advantage across by 8x.
(4) It is what Policy Decorator does - their residual's action space is
literally single_action_space.low.repeat(act_horizon).

Mean rather than sum over the sub-steps is a units choice, not a semi-MDP
subtlety: the chunk length is constant, so the two differ by exactly a factor
of act_horizon. The mean keeps rewards in [0,1] and value targets O(1) with no
reward scaling. The sub-step loop still divides by the number of steps ACTUALLY
executed, because a silent partial chunk would be a units bug rather than a
crash.

    T3_RUN=t3/artifacts/gap_gen2 MODE=gap python t4/train_ppo.py --seed 1
"""
import json
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import env_t4                                    # noqa: E402,F401  registers the env
from make_envs import make_train_envs            # noqa: E402
from residual import (ACT_DIM, RES_DIM, ResidualAgent, ResidualHead,  # noqa: E402
                      alpha_at, apply_delta, save_residual)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "t2"))
from harness import inspect_ckpt, load_weights, manifest, sha16  # noqa: E402


@dataclass
class Args:
    exp_name: Optional[str] = None
    seed: int = 1
    """the residual's training seed. T-IV pairs residual seed b with evaluation
    block b, so this is also which block the result is reported in."""
    mode: str = "gap"
    """the T-II failure mode being targeted; recorded in the checkpoint."""
    ckpt: str = ""
    """the FROZEN base policy. Defaults to the committed one."""
    out: str = ""
    """run directory. Defaults to t4/runs/<mode>."""
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "RIPL-T4"
    wandb_entity: Optional[str] = None

    # -- environment --------------------------------------------------------
    env_id: str = "StackCube-T4-v1"
    control_mode: str = "pd_ee_delta_pos"
    obs_mode: str = "rgb"
    sim_backend: str = "physx_cuda"
    max_episode_steps: int = 200
    num_envs: int = 64

    # -- the residual -------------------------------------------------------
    alpha: float = 0.05
    """PD's res_scale. pd_ee_delta_pos maps +-1 to +-0.1 m, so 0.05 = 5 mm/step."""
    alpha_warmup: int = 1_000_000
    """H, in ENV steps. PD's progressive exploration schedule, ramping the bound
    instead of the mixing probability - see alpha_at() in residual.py."""
    res_horizon: int = 0
    """how many env steps one residual action covers. 0 = the base's act_horizon
    (PD's shape). Smaller re-queries the residual against a fresher observation
    WITHOUT re-planning the frozen base more often. Must divide act_horizon."""
    hidden: int = 256
    log_std_init: float = -1.0

    # -- PPO ----------------------------------------------------------------
    total_timesteps: int = 4_000_000
    """in ENV steps, so it is comparable to the sim cost and to PD's numbers."""
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    num_steps: int = 0
    """residual steps per iteration. 0 = auto = exactly one episode
    (max_episode_steps / res_horizon), which is 25 at act_horizon 8. Auto
    because the right value MOVES with --res-horizon, and an iteration that
    straddles an episode boundary makes every per-iteration curve harder to
    read for no gain."""
    gamma: float = 0.9
    """PER CHUNK; ~0.987 per env step at act_horizon 8."""
    gae_lambda: float = 0.95
    num_minibatches: int = 8
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = 0.1
    finite_horizon_gae: bool = False

    log_freq: int = 1
    save_freq: int = 50

    batch_size: int = field(init=False, default=0)
    minibatch_size: int = field(init=False, default=0)
    num_iterations: int = field(init=False, default=0)


class Logger:
    """wandb AND a local CSV. The CSV is not redundant: figures/ is committed
    and pulled off the pod, wandb needs a login and a network, and CLAUDE.md is
    explicit that anything not pulled before the pod stops is gone."""

    def __init__(self, path, use_wandb=False):
        self.rows, self.path, self.wandb = [], path, use_wandb
        self.cur = {}

    def add(self, tag, value, step):
        self.cur[tag] = float(value)
        self.cur["global_step"] = int(step)

    def flush(self):
        if not self.cur:
            return
        self.rows.append(dict(self.cur))
        if self.wandb:
            import wandb
            wandb.log(self.cur, step=self.cur["global_step"])
        self.cur = {}
        keys = sorted({k for r in self.rows for k in r})
        import csv
        with open(self.path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.rows)


def index_obs(obs, mask):
    if isinstance(obs, dict):
        return {k: index_obs(v, mask) for k, v in obs.items()}
    return obs[mask]


def main():
    args = tyro.cli(Args)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    args.ckpt = args.ckpt or os.path.join(
        root, "checkpoints", "stackcube_rgb_spatial_800demos.pt")
    args.out = args.out or os.path.join(root, "t4", "runs", args.mode)
    os.makedirs(args.out, exist_ok=True)
    run_name = args.exp_name or f"{args.mode}_seed{args.seed}"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    print(env_t4.describe())
    if os.environ.get("T3_SAMPLER", "1") in ("0", "false", "False", ""):
        print("!! T3_SAMPLER is OFF, so this trains on the NOMINAL distribution.\n"
              "   That is not what T-IV asks for; unset it unless you meant it.")

    # ---------------------------------------------------------------- envs
    envs = make_train_envs(args.env_id, args.num_envs,
                           max_episode_steps=args.max_episode_steps,
                           control_mode=args.control_mode, obs_mode=args.obs_mode,
                           reward_mode="normalized_dense",
                           sim_backend=args.sim_backend)

    # -------------------------------------------------------- frozen base
    # Built exactly as t2/harness.build_agent builds it, against the same
    # observation stack, so the policy being decorated here is the policy
    # measured there. The encoder variant is inferred from the weights.
    sd, is_rgb, pooled = inspect_ckpt(args.ckpt, args.obs_mode == "state")
    dp = f"{os.environ['MANISKILL_REPO']}/examples/baselines/diffusion_policy"
    if dp not in sys.path:
        sys.path.insert(0, dp)
    if args.obs_mode == "state":
        import train as T
    else:
        import train_rgbd as T
    import dataclasses
    kw = dict(env_id=args.env_id, demo_path="unused.h5",
              control_mode=args.control_mode, sim_backend=args.sim_backend,
              max_episode_steps=args.max_episode_steps)
    if is_rgb:
        kw["obs_mode"] = "rgb"
        if any(f.name == "pool_feature_map" for f in dataclasses.fields(T.Args)):
            kw["pool_feature_map"] = pooled
        elif not pooled:
            sys.exit("\nspatial-encoder checkpoint against a stock ManiSkill; "
                     "run 'bash setup/apply_patches.sh' first.\n")
    dp_args = T.Args(**kw)
    base = load_weights(T.Agent(envs, dp_args).to(device), sd)
    for p in base.parameters():
        p.requires_grad_(False)

    act_horizon = dp_args.act_horizon
    res_horizon = args.res_horizon or act_horizon
    if act_horizon % res_horizon:
        sys.exit(f"\n--res-horizon {res_horizon} does not divide the base "
                 f"policy's act_horizon {act_horizon}; the residual would "
                 f"straddle two base plans.\n")
    steps_per_ep = args.max_episode_steps // res_horizon
    if args.num_steps <= 0:
        args.num_steps = steps_per_ep
    if args.num_steps % steps_per_ep:
        print(f"!! --num-steps {args.num_steps} is not a multiple of the "
              f"{steps_per_ep} residual steps in an episode; iterations will "
              f"straddle episode boundaries. GAE still handles it, but the "
              f"per-iteration curves get harder to read.")

    agent = ResidualAgent(base, head=None, act_horizon=act_horizon)
    obs, _ = envs.reset(seed=args.seed)
    emb_dim = agent.emb_dim(obs)
    head = ResidualHead(emb_dim, res_horizon=res_horizon, hidden=args.hidden,
                        log_std_init=args.log_std_init).to(device)
    agent.set_residual(head, args.alpha)
    n_par = sum(p.numel() for p in head.parameters())

    # env steps per residual step, per iteration
    env_per_iter = args.num_envs * args.num_steps * res_horizon
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = max(1, args.total_timesteps // env_per_iter)

    meta = manifest(ckpt=os.path.abspath(args.ckpt), mode=args.mode,
                    seed=args.seed, t3_run=os.environ.get("T3_RUN", ""),
                    t3_sampler=int(os.environ.get("T3_SAMPLER", "1") not in
                                   ("0", "false", "False", "")),
                    t4_nominal_frac=env_t4.nominal_frac(),
                    alpha=args.alpha, alpha_warmup=args.alpha_warmup,
                    act_horizon=act_horizon, res_horizon=res_horizon,
                    emb_dim=emb_dim, head_params=n_par,
                    num_envs=args.num_envs, num_steps=args.num_steps,
                    total_timesteps=args.total_timesteps,
                    num_iterations=args.num_iterations)
    for k, v in meta.items():
        print(f"  {k:18s} {v}")
    print(f"  residual head      {n_par:,} params over a {emb_dim}-d embedding")
    print(f"  one iteration      {args.num_steps} residual steps x {args.num_envs} "
          f"envs = {env_per_iter:,} env steps")

    if args.track:
        import wandb
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity,
                   name=run_name, config=dict(vars(args), **meta), save_code=True,
                   group=f"t4-{args.mode}", tags=["t4", "residual", "ppo"])
    logger = Logger(os.path.join(args.out, f"{run_name}_train.csv"), args.track)

    optimizer = optim.Adam(head.parameters(), lr=args.learning_rate, eps=1e-5)

    # ------------------------------------------------------------- storage
    S, E = args.num_steps, args.num_envs
    b_emb = torch.zeros((S, E, emb_dim), device=device)
    b_act = torch.zeros((S, E, head.out_dim), device=device)
    b_logp = torch.zeros((S, E), device=device)
    b_rew = torch.zeros((S, E), device=device)
    b_done = torch.zeros((S, E), device=device)
    b_val = torch.zeros((S, E), device=device)

    global_step = 0                       # ENV steps, not residual steps
    start = time.time()
    next_done = torch.zeros(E, device=device)
    vram = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for it in range(1, args.num_iterations + 1):
        # alpha is constant across this whole collect+update. It does NOT enter
        # the log-prob - the raw Gaussian sample is the action and alpha*tanh is
        # part of the transition - so what would break if it moved mid-iteration
        # is the MDP itself: returns and value targets would describe two
        # different environments. Between iterations it is fine; PPO tolerates
        # slow non-stationarity, which is the whole idea of the schedule.
        alpha = alpha_at(global_step, args.alpha, args.alpha_warmup)
        agent.alpha = alpha
        if args.anneal_lr:
            optimizer.param_groups[0]["lr"] = (
                (1.0 - (it - 1.0) / args.num_iterations) * args.learning_rate)

        head.eval()
        final_values = torch.zeros((S, E), device=device)
        ep = defaultdict(list)
        t_roll = time.time()
        dnorm = []

        for step in range(S):
            with torch.no_grad():
                b_emb_t = agent.embed(obs)
            b_emb[step] = b_emb_t
            b_done[step] = next_done
            with torch.no_grad():
                raw, logp, _, val = head.get_action_and_value(b_emb_t)
            b_act[step], b_logp[step], b_val[step] = raw, logp, val.flatten()

            with torch.no_grad():
                chunk = agent.base_chunk(obs)
                exe = apply_delta(chunk, raw, alpha)
            dnorm.append(float((exe[..., :RES_DIM] - chunk[..., :RES_DIM])
                               .norm(dim=-1).mean()) * 100.0)   # mm

            # ---- execute the chunk -------------------------------------
            rsum, n_sub, done_mid = 0.0, 0, False
            for i in range(exe.shape[1]):
                obs, rew, term, trunc, infos = envs.step(exe[:, i])
                rsum = rsum + rew.view(-1)
                n_sub += 1
                global_step += E
                if bool(torch.logical_or(term, trunc).any()):
                    done_mid = True
                    break
            b_rew[step] = rsum / n_sub          # mean over the steps EXECUTED
            next_done = torch.logical_or(term, trunc).to(torch.float32)

            if done_mid:
                # The env auto-reset under us, so any unconsumed base plan
                # belongs to the finished episode.
                agent.reset_chunk()
            if "final_info" in infos:
                m = infos["_final_info"]
                for k, v in infos["final_info"]["episode"].items():
                    ep[k].append(v[m].float().mean().item())
                with torch.no_grad():
                    fo = index_obs(infos["final_observation"], m)
                    final_values[step, torch.arange(E, device=device)[m]] = \
                        head.get_value(agent.embed(fo)).view(-1)
        t_roll = time.time() - t_roll

        # ---- GAE (upstream's, verbatim in structure) --------------------
        with torch.no_grad():
            next_value = head.get_value(agent.embed(obs)).reshape(1, -1)  # noqa: E501
            adv = torch.zeros_like(b_rew)
            lastgaelam = 0
            for t in reversed(range(S)):
                if t == S - 1:
                    next_not_done, nextvalues = 1.0 - next_done, next_value
                else:
                    next_not_done, nextvalues = 1.0 - b_done[t + 1], b_val[t + 1]
                real_next = next_not_done * nextvalues + final_values[t]
                delta = b_rew[t] + args.gamma * real_next - b_val[t]
                adv[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam)
            ret = adv + b_val

        f_emb = b_emb.reshape(-1, emb_dim)
        f_act = b_act.reshape(-1, head.out_dim)
        f_logp, f_adv, f_ret, f_val = (b_logp.reshape(-1), adv.reshape(-1),
                                       ret.reshape(-1), b_val.reshape(-1))

        # ---- update ------------------------------------------------------
        head.train()
        inds = np.arange(args.batch_size)
        clipfracs, t_upd = [], time.time()
        approx_kl = torch.zeros((), device=device)
        for _ in range(args.update_epochs):
            np.random.shuffle(inds)
            for s0 in range(0, args.batch_size, args.minibatch_size):
                mb = inds[s0:s0 + args.minibatch_size]
                _, newlogp, entropy, newval = head.get_action_and_value(
                    f_emb[mb], f_act[mb])
                logratio = newlogp - f_logp[mb]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item())
                if args.target_kl is not None and approx_kl > args.target_kl:
                    break
                mba = f_adv[mb]
                if args.norm_adv:
                    mba = (mba - mba.mean()) / (mba.std() + 1e-8)
                pg = torch.max(-mba * ratio,
                               -mba * torch.clamp(ratio, 1 - args.clip_coef,
                                                  1 + args.clip_coef)).mean()
                newval = newval.view(-1)
                if args.clip_vloss:
                    unc = (newval - f_ret[mb]) ** 2
                    cl = f_val[mb] + torch.clamp(newval - f_val[mb],
                                                 -args.clip_coef, args.clip_coef)
                    v_loss = 0.5 * torch.max(unc, (cl - f_ret[mb]) ** 2).mean()
                else:
                    v_loss = 0.5 * ((newval - f_ret[mb]) ** 2).mean()
                ent = entropy.mean()
                loss = pg - args.ent_coef * ent + v_loss * args.vf_coef
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(head.parameters(), args.max_grad_norm)
                optimizer.step()
            if args.target_kl is not None and approx_kl > args.target_kl:
                break
        t_upd = time.time() - t_upd

        yv = f_val.cpu().numpy(); yt = f_ret.cpu().numpy()
        var_y = np.var(yt)
        expl = float("nan") if var_y == 0 else 1 - np.var(yt - yv) / var_y
        vram = (torch.cuda.max_memory_allocated() / 2**30
                if device.type == "cuda" else 0.0)

        if it % args.log_freq == 0 or it == args.num_iterations:
            for k, v in ep.items():
                logger.add(f"train/{k}", float(np.mean(v)), global_step)
            logger.add("charts/alpha", alpha, global_step)
            logger.add("charts/alpha_mm", alpha * 100.0, global_step)
            logger.add("charts/delta_norm_mm", float(np.mean(dnorm)), global_step)
            logger.add("charts/learning_rate",
                       optimizer.param_groups[0]["lr"], global_step)
            logger.add("charts/reward_mean", float(b_rew.mean()), global_step)
            logger.add("losses/policy_loss", pg.item(), global_step)
            logger.add("losses/value_loss", v_loss.item(), global_step)
            logger.add("losses/entropy", ent.item(), global_step)
            logger.add("losses/approx_kl", approx_kl.item(), global_step)
            logger.add("losses/clipfrac", float(np.mean(clipfracs)), global_step)
            logger.add("losses/explained_variance", expl, global_step)
            logger.add("sys/vram_max_gb", vram, global_step)
            logger.add("sys/wall_seconds", time.time() - start, global_step)
            logger.add("charts/SPS", global_step / (time.time() - start), global_step)
            logger.add("time/rollout", t_roll, global_step)
            logger.add("time/update", t_upd, global_step)
            logger.add("iteration", it, global_step)
            logger.flush()
            so = np.mean(ep.get("success_once", [float("nan")]))
            print(f"it {it:4d}/{args.num_iterations}  step {global_step:>9,}  "
                  f"success_once {so:.3f}  rew {float(b_rew.mean()):.4f}  "
                  f"alpha {alpha * 100:.2f}mm  |d| {np.mean(dnorm):.2f}mm  "
                  f"kl {approx_kl.item():.4f}  vram {vram:.1f}G  "
                  f"{global_step / (time.time() - start):.0f} env-step/s",
                  flush=True)

        if it % args.save_freq == 0 or it == args.num_iterations:
            save_residual(os.path.join(args.out, f"residual_seed{args.seed}.pt"),
                          head, args.alpha, act_horizon, mode=args.mode,
                          seed=args.seed, base_sha256=sha16(args.ckpt),
                          iteration=it, global_step=global_step,
                          alpha_warmup=args.alpha_warmup)

    if global_step < args.alpha_warmup:
        print(f"\n!! training stopped at {global_step:,} env steps but "
              f"--alpha-warmup is {args.alpha_warmup:,}, so the bound only ever "
              f"reached {alpha * 100:.2f} mm of the requested "
              f"{args.alpha * 100:.2f} mm.\n   The saved checkpoint records the "
              f"FULL alpha, so evaluation would run it at a bound it was never "
              f"trained at.\n   Lower --alpha-warmup or raise "
              f"--total-timesteps.")

    meta["wall_seconds"] = round(time.time() - start, 1)
    meta["vram_max_gb"] = round(vram, 2)
    meta["env_steps"] = global_step
    with open(os.path.join(args.out, f"{run_name}_manifest.json"), "w") as f:
        json.dump(meta, f, indent=2)
    envs.close()
    print(f"\ndone: {global_step:,} env steps in {meta['wall_seconds']:.0f}s, "
          f"peak VRAM {vram:.1f} GB")
    print(f"  {os.path.join(args.out, f'residual_seed{args.seed}.pt')}")


if __name__ == "__main__":
    main()
