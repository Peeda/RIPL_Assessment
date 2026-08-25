#!/usr/bin/env python
"""Prove the rollouts are driven by THESE weights, and not by something else.

verify.py establishes offline that the episodes are the seeds that were asked
for, in the region that was asked for - by joining the logged initial states
against a separately generated, policy-free seed index. What it cannot see is
whether the actions came from the checkpoint. A harness that silently fell back
to random actions, or loaded an untrained network, would still produce a CSV
whose initial states cross-check perfectly.

So run the same seeds four ways and compare:

  policy      the checkpoint, as the real passes use it
  untrained   identical architecture, freshly initialised weights
  random      uniform samples from the action space
  zero        no action at all

The claim "the rollouts are driven by the trained policy" is exactly the claim
that arm 1 beats the other three by a wide margin. 'untrained' is the arm that
matters most: random and zero only show that *some* network is being consulted,
while untrained shows it is the LEARNED weights doing the work.

  python t2/policy_check.py $CKPT --episodes 30 --seeds 6000
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t2_common import build_agent, manifest, to_device, wilson  # noqa: E402

MAX_EP_STEPS = 200


def run(envs, act_fn, seeds, num_envs, device):
    """One arm. Returns success_once per episode, in seed order."""
    out = []
    for start in range(0, len(seeds), num_envs):
        batch = seeds[start:start + num_envs]
        bseeds = batch + [batch[-1]] * (num_envs - len(batch))
        obs, info = envs.reset(seed=bseeds)
        ever = np.zeros(num_envs, bool)
        steps, done = 0, False
        while not done and steps < MAX_EP_STEPS:
            chunk = act_fn(obs)
            for i in range(chunk.shape[1]):
                obs, rew, term, trunc, info = envs.step(chunk[:, i])
                steps += 1
                if "success" in info:
                    ever |= np.asarray(
                        [bool(np.asarray(x).reshape(-1)[0])
                         for x in np.atleast_1d(info["success"])], bool)
                if np.any(trunc) or np.any(term):
                    done = True
                    break
        out += list(ever[:len(batch)])
    return np.array(out, dtype=int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=6000,
                    help="first seed. Default 6000 - above the demonstration "
                         "range, so this is not measuring memorisation either.")
    ap.add_argument("--num-envs", type=int, default=10)
    ap.add_argument("--state", action="store_true")
    a = ap.parse_args()

    seeds = list(range(a.seeds, a.seeds + a.episodes))
    agent, envs, args, device = build_agent(a.ckpt, a.state, a.num_envs,
                                            max_episode_steps=MAX_EP_STEPS)
    meta = manifest(ckpt=os.path.abspath(a.ckpt), episodes=a.episodes,
                    seed_min=min(seeds), seed_max=max(seeds))
    print(f"  checkpoint  {os.path.basename(a.ckpt)}")
    print(f"  sha256[:16] {meta.get('ckpt_sha256')}   <- quote this next to any number")
    print(f"  seeds       {min(seeds)}..{max(seeds)}  ({a.episodes} episodes per arm)")
    print("")

    # Same architecture, never loaded. type(agent) avoids re-importing the
    # baseline's module and guarantees it is literally the same class.
    untrained = type(agent)(envs, args).to(device)

    def policy(obs):
        with torch.no_grad():
            return agent.get_action(to_device(obs, device)).cpu().numpy()

    def untrained_act(obs):
        with torch.no_grad():
            return untrained.get_action(to_device(obs, device)).cpu().numpy()


    # Chunk shape is PROBED from the policy rather than read off Args, so every
    # arm steps the env exactly as many times as the policy arm does. A control
    # that stepped a different number of times would not be a control.
    obs0, _ = envs.reset(seed=seeds[:1] * a.num_envs)
    with torch.no_grad():
        H, A = policy(obs0).shape[1:]
    print(f"  action chunk  {H} steps x {A} dims per policy call")

    def random_act(obs):
        return np.stack([np.stack([envs.single_action_space.sample()
                                   for _ in range(H)])
                         for _ in range(a.num_envs)]).astype(np.float32)

    def zero_act(obs):
        return np.zeros((a.num_envs, H, A), dtype=np.float32)

    arms = [("policy", policy), ("untrained", untrained_act),
            ("random", random_act), ("zero", zero_act)]
    res = {}
    for name, fn in arms:
        torch.manual_seed(1)
        np.random.seed(1)
        envs.action_space.seed(1)
        res[name] = run(envs, fn, seeds, a.num_envs, device)
        k, n = int(res[name].sum()), len(res[name])
        lo, hi = wilson(k, n)
        print(f"  {name:<10} success_once {k}/{n} = {k/n:.3f}  [{lo:.3f}, {hi:.3f}]",
              flush=True)
    envs.close()

    p = res["policy"].mean()
    others = max(res[k].mean() for k in ("untrained", "random", "zero"))
    print("")
    if p > others + 0.20:
        print(f"  PASS  the policy arm beats every control by "
              f"{p - others:+.3f}. The rollouts are driven by the trained")
        print(f"        weights - not by the harness, and not by the architecture alone.")
    else:
        print(f"  FAIL  the policy arm is only {p - others:+.3f} above the best control.")
        print(f"        Either the weights are not being loaded, the actions are not")
        print(f"        reaching the env, or the checkpoint is not what it claims.")
        sys.exit(1)


if __name__ == "__main__":
    main()
