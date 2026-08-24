#!/usr/bin/env python
"""Does the trained RGB policy actually look at the image?

    source /workspace/ripl/env.sh
    python probe_visual_dependence.py <checkpoint.pt>

Diagnoses one specific failure: a visual policy that has learned the *shape* of
the demonstrated motion (descend to table height, close gripper) but not where
to aim it, and so replays an average trajectory regardless of what it sees.
From the eval videos that looks like "reaches confidently, grabs at nothing".

Method: hold proprioception fixed and swap only the image between observations
drawn from different initial states. If the predicted action barely moves, the
visual encoder is being ignored and no amount of further training on the same
data will fix it. If it moves a lot, perception works and the problem is
precision.

Read-only. Does not touch the training run.
"""
import os, sys, torch, numpy as np, gymnasium as gym, mani_skill.envs

DP = f"{os.environ['MANISKILL_REPO']}/examples/baselines/diffusion_policy"
sys.path.insert(0, DP)
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from diffusion_policy.make_env import make_eval_envs
import train_rgbd as T

CKPT = sys.argv[1]
N_STATES = 8


def main():
    args = T.Args(
        env_id=os.environ.get("ENV_ID", "StackCube-v1"),
        demo_path="unused.h5",
        control_mode=os.environ.get("CTRL", "pd_ee_delta_pos"),
        sim_backend=os.environ.get("BACKEND", "physx_cpu"),
        obs_mode="rgb",
        max_episode_steps=200,
    )
    env_kwargs = dict(control_mode=args.control_mode, reward_mode="sparse",
                      obs_mode="rgb", render_mode="rgb_array",
                      human_render_camera_configs=dict(shader_pack="default"),
                      max_episode_steps=200)
    envs = make_eval_envs(args.env_id, 1, args.sim_backend, env_kwargs,
                          dict(obs_horizon=args.obs_horizon), video_dir=None,
                          wrappers=[FlattenRGBDObservationWrapper])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = T.Agent(envs, args).to(device)
    sd = torch.load(CKPT, map_location=device)
    agent.load_state_dict(sd.get("ema_agent", sd.get("agent")))
    agent.eval()

    # Collect observations from distinct initial states.
    obs_list = []
    for i in range(N_STATES):
        o, _ = envs.reset(seed=1000 + i)
        obs_list.append({k: torch.as_tensor(np.asarray(v)).to(device)
                         for k, v in o.items()})

    def act(o):
        with torch.no_grad():
            a = agent.get_action({k: v.clone() for k, v in o.items()})
        return a[0, 0].float().cpu().numpy()   # first action of the chunk

    base = act(obs_list[0])

    # (1) how much does the action vary across genuinely different scenes?
    acts = np.stack([act(o) for o in obs_list])
    spread_all = acts.std(axis=0)

    # (2) hold proprioception fixed, swap ONLY the image
    swapped = []
    for o in obs_list[1:]:
        mixed = {k: v.clone() for k, v in obs_list[0].items()}
        mixed["rgb"] = o["rgb"].clone()
        swapped.append(act(mixed))
    swapped = np.stack([base] + swapped)
    spread_img = swapped.std(axis=0)

    envs.close()

    np.set_printoptions(precision=4, suppress=True)
    print(f"\ncheckpoint: {os.path.basename(CKPT)}")
    print(f"action dims: {len(base)}  (3 translation + gripper)\n")
    print(f"  action std across {N_STATES} full scenes     : {spread_all}")
    print(f"  action std varying ONLY the image        : {spread_img}")

    t_all = float(np.linalg.norm(spread_all[:3]))
    t_img = float(np.linalg.norm(spread_img[:3]))
    print(f"\n  translation spread, full scenes : {t_all:.4f}")
    print(f"  translation spread, image only  : {t_img:.4f}")
    if t_all > 1e-9:
        print(f"  fraction attributable to vision : {t_img / t_all:.1%}")

    print("")
    if t_img < 0.02:
        print("VERDICT: the policy is effectively BLIND. Swapping the image barely")
        print("moves the action, so it is replaying an average trajectory. More")
        print("iterations on this data will not help; more demonstrations might.")
    elif t_img < 0.1 * max(t_all, 1e-9):
        print("VERDICT: vision contributes weakly. Mostly proprioception-driven.")
    else:
        print("VERDICT: the action does depend on the image. Perception works;")
        print("the failure is precision, not blindness.")


if __name__ == "__main__":
    main()
