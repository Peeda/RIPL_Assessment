#!/usr/bin/env python
"""Roll out a checkpoint and log per-episode initial states + outcomes.

    source /workspace/ripl/env.sh
    python rollout_log.py <checkpoint.pt> [n_episodes] [--state]

This is the T-II harness. It also answers the immediate question the eval
videos raise: does the arm's reach target follow the red cube, or does it go to
the same place every time?

Per episode it writes a row of the schema fixed in CLAUDE.md:

    seed, cubeA_x/y/theta, cubeB_x/y/theta, separation, relative_yaw,
    ep_len, success, is_cubeA_grasped, is_cubeA_on_cubeB, is_cubeA_static

plus tcp_x/tcp_y at the moment of deepest descent - the policy's chosen reach
target. Correlating that against cubeA_x/y is the direct test of localisation:
r near 1 means the policy aims at the cube, r near 0 means it aims at the mean
of the training set no matter what it sees.

All angles wrapped to (-pi, pi] at log time, once, per CLAUDE.md.

num_envs=1 on purpose: make_eval_envs uses SyncVectorEnv at N=1, which keeps the
env in-process so ground-truth cube poses are reachable. Under AsyncVectorEnv
they are not.
"""
import os, sys, math, csv, dataclasses, torch, numpy as np, gymnasium as gym, mani_skill.envs

DP = f"{os.environ['MANISKILL_REPO']}/examples/baselines/diffusion_policy"
sys.path.insert(0, DP)
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from diffusion_policy.make_env import make_eval_envs

CKPT = sys.argv[1]
N_EP = int(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith('-') else 50
STATE_MODE = '--state' in sys.argv
OUT = os.path.splitext(os.path.basename(CKPT))[0] + '_rollouts.csv'


def wrap(a):
    """(-pi, pi]. The one convention; see CLAUDE.md."""
    return -((-a + math.pi) % (2 * math.pi) - math.pi)


FLAG_KEYS = ('success_once', 'success_at_end', 'success',
             'is_cubeA_grasped', 'is_cubeA_on_cubeB', 'is_cubeA_static')


def flag(d, k):
    """One info entry as 0/1, or '' when the key is absent."""
    v = d.get(k) if isinstance(d, dict) else None
    return int(bool(np.asarray(v).reshape(-1)[0])) if v is not None else ''


def yaw(q):
    w, x, y, z = [float(v) for v in np.asarray(q).reshape(-1)[:4]]
    return wrap(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def check_mode_matches(ckpt_path):
    """Fail clearly on the easy mistake: rgb checkpoint with --state, or vice
    versa. Without this the error is a wall of shape mismatches that buries the
    one-line cause."""
    sd = torch.load(ckpt_path, map_location='cpu')
    keys = (sd.get('ema_agent') or sd.get('agent')).keys()
    ckpt_is_rgb = any(k.startswith('visual_encoder.') for k in keys)
    if ckpt_is_rgb and STATE_MODE:
        sys.exit(f"\n{os.path.basename(ckpt_path)} is an RGB checkpoint (it has "
                 f"visual_encoder weights) but --state was passed.\n"
                 f"Drop --state, or point at the state run's checkpoint.\n")
    if not ckpt_is_rgb and not STATE_MODE:
        sys.exit(f"\n{os.path.basename(ckpt_path)} is a state checkpoint (no "
                 f"visual_encoder weights) but --state was not passed.\n"
                 f"Add --state, or point at the rgb run's checkpoint.\n")
    print(f"  checkpoint mode: {'rgb' if ckpt_is_rgb else 'state'}  (matches args)")

    # Infer the encoder variant from the weights rather than trusting a flag
    # the caller has to remember. visual_encoder.fc.0.weight is (256, 128) when
    # PlainConv global-max-pooled its feature map and (256, 128*8*8) when it
    # kept the 8x8 grid. Getting this wrong is not a wrong number, it is a
    # shape mismatch at load_state_dict - but the message names 8192 vs 128
    # rather than the cause, so name the cause here.
    pooled = None
    if ckpt_is_rgb:
        w = (sd.get('ema_agent') or sd.get('agent'))['visual_encoder.fc.0.weight']
        pooled = w.shape[1] == 128
        print(f"  visual encoder:  {'pooled (global max, no spatial map)' if pooled else 'spatial (8x8 map kept)'}"
              f"  [fc in={w.shape[1]}]")
    return sd, pooled


def main():
    ckpt_sd, ckpt_pooled = check_mode_matches(CKPT)
    if STATE_MODE:
        import train as T
        obs_mode, wrappers = 'state', []
    else:
        import train_rgbd as T
        obs_mode, wrappers = 'rgb', [FlattenRGBDObservationWrapper]

    env_id = os.environ.get('ENV_ID', 'StackCube-v1')
    ctrl = os.environ.get('CTRL', 'pd_ee_delta_pos')
    backend = os.environ.get('BACKEND', 'physx_cpu')

    kw = dict(env_id=env_id, demo_path='unused.h5', control_mode=ctrl,
              sim_backend=backend, max_episode_steps=200)
    if not STATE_MODE:
        kw['obs_mode'] = 'rgb'
        # Only present once patches/0001 is applied; on a stock checkout the
        # encoder is hardwired to the pooled variant and there is nothing to set.
        if any(f.name == 'pool_feature_map' for f in dataclasses.fields(T.Args)):
            kw['pool_feature_map'] = ckpt_pooled
        elif not ckpt_pooled:
            sys.exit("\nThis is a spatial-encoder checkpoint but "
                     f"{os.environ['MANISKILL_REPO']} is stock upstream.\n"
                     "Run 'bash apply_patches.sh' first.\n")
    args = T.Args(**kw)

    env_kwargs = dict(control_mode=ctrl, reward_mode='sparse', obs_mode=obs_mode,
                      render_mode='rgb_array',
                      human_render_camera_configs=dict(shader_pack='default'),
                      max_episode_steps=200)
    envs = make_eval_envs(env_id, 1, backend, env_kwargs,
                          dict(obs_horizon=args.obs_horizon), video_dir=None,
                          wrappers=wrappers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent = T.Agent(envs, args).to(device)
    # ema_agent is what train.py evaluates with, so it is what produced the
    # reported success numbers. Prefer it over the raw agent weights.
    agent.load_state_dict(ckpt_sd.get('ema_agent') or ckpt_sd.get('agent'))
    agent.eval()

    base = envs.envs[0].unwrapped          # SyncVectorEnv keeps this reachable
    rows = []

    # Write per episode rather than once at the end. A 100-episode rollout is
    # long enough that it will sometimes be interrupted, and CLAUDE.md's rule is
    # to have the initial states and flags for every rollout - partial data is
    # worth far more than an empty file.
    csv_f = open(OUT, 'w', newline='')
    csv_w = None

    for ep in range(N_EP):
        seed = 10_000 + ep
        obs, _ = envs.reset(seed=seed)
        a_p, a_q = base.cubeA.pose.p[0], base.cubeA.pose.q[0]
        b_p, b_q = base.cubeB.pose.p[0], base.cubeB.pose.q[0]
        ax, ay = float(a_p[0]), float(a_p[1])
        bx, by = float(b_p[0]), float(b_p[1])
        row = dict(seed=seed,
                   cubeA_x=ax, cubeA_y=ay, cubeA_theta=yaw(a_q),
                   cubeB_x=bx, cubeB_y=by, cubeB_theta=yaw(b_q),
                   separation=math.dist((ax, ay), (bx, by)),
                   relative_yaw=wrap(yaw(b_q) - yaw(a_q)))

        lowest_z, tcp_at_lowest, steps = 1e9, (float('nan'),) * 2, 0
        flags = {k: '' for k in FLAG_KEYS}
        done = False
        while not done and steps < 200:
            o = {k: torch.as_tensor(np.asarray(v)).to(device)
                 for k, v in obs.items()} if isinstance(obs, dict) else \
                torch.as_tensor(np.asarray(obs)).to(device)
            with torch.no_grad():
                chunk = agent.get_action(o)
            chunk = chunk.cpu().numpy() if backend == 'physx_cpu' else chunk
            for i in range(chunk.shape[1]):
                obs, rew, term, trunc, info = envs.step(chunk[:, i])
                steps += 1
                tcp = base.agent.tcp.pose.p[0]
                if float(tcp[2]) < lowest_z:
                    lowest_z = float(tcp[2])
                    tcp_at_lowest = (float(tcp[0]), float(tcp[1]))
                if trunc.any() or term.any():
                    done = True
                    fi = info.get('final_info', info)
                    fi = fi[0] if isinstance(fi, (list, np.ndarray)) else fi

                    # make_eval_envs builds CPUGymWrapper(ignore_terminations=
                    # True, record_metrics=True), so every episode runs the full
                    # horizon and the metrics land NESTED under info['episode'],
                    # not at the top level. The top-level 'success' is therefore
                    # the final-step value - success_at_end - and is NOT the
                    # success_once that CLAUDE.md fixes as the reported T-I
                    # number and that wandb's eval/success_once logs. Read both.
                    epi = fi.get('episode', {}) if isinstance(fi, dict) else {}
                    flags['success_once'] = flag(epi, 'success_once')
                    flags['success_at_end'] = flag(epi, 'success_at_end')
                    for k in ('success', 'is_cubeA_grasped',
                              'is_cubeA_on_cubeB', 'is_cubeA_static'):
                        flags[k] = flag(fi, k)
                    break

        row.update(tcp_x=tcp_at_lowest[0], tcp_y=tcp_at_lowest[1],
                   tcp_lowest_z=lowest_z, ep_len=steps, **flags)
        rows.append(row)
        if csv_w is None:
            csv_w = csv.DictWriter(csv_f, fieldnames=list(row.keys()))
            csv_w.writeheader()
        csv_w.writerow(row)
        csv_f.flush()
        # flush the console too: piping through tee makes stdout block-buffered,
        # which looks exactly like a hang for the first several minutes.
        print(f"  ep {ep:>3} seed {seed}  cubeA=({ax:+.3f},{ay:+.3f})  "
              f"tcp=({tcp_at_lowest[0]:+.3f},{tcp_at_lowest[1]:+.3f})  "
              f"success_once={flags.get('success_once','?')}", flush=True)

    envs.close()
    csv_f.close()
    print(f"\nwrote {OUT}  ({len(rows)} episodes)")

    # --- the localisation test -------------------------------------------
    def corr(u, v):
        u, v = np.asarray(u, float), np.asarray(v, float)
        m = np.isfinite(u) & np.isfinite(v)
        if m.sum() < 3 or u[m].std() < 1e-9 or v[m].std() < 1e-9:
            return float('nan')
        return float(np.corrcoef(u[m], v[m])[0, 1])

    ax_ = [r['cubeA_x'] for r in rows]; ay_ = [r['cubeA_y'] for r in rows]
    tx_ = [r['tcp_x'] for r in rows];   ty_ = [r['tcp_y'] for r in rows]
    print(f"\n  cubeA_x spread {np.std(ax_):.4f} m   reach tcp_x spread {np.nanstd(tx_):.4f} m")
    print(f"  cubeA_y spread {np.std(ay_):.4f} m   reach tcp_y spread {np.nanstd(ty_):.4f} m")
    print(f"  corr(cubeA_x, tcp_x) = {corr(ax_, tx_):+.3f}")
    print(f"  corr(cubeA_y, tcp_y) = {corr(ay_, ty_):+.3f}")
    err = [math.dist((r['cubeA_x'], r['cubeA_y']), (r['tcp_x'], r['tcp_y']))
           for r in rows if np.isfinite(r['tcp_x'])]
    print(f"  mean reach error to cubeA: {np.mean(err)*1000:.0f} mm "
          f"(cube is 40 mm wide)")
    # success_once is the T-I number and what wandb's eval/success_once reports;
    # success_at_end is the same episode judged at step 200. A large gap is not
    # noise - it is cubeA stacked and then toppling, a T-II mode in its own
    # right, distinct from the separation one. See CLAUDE.md.
    def rate(k):
        v = [r.get(k) for r in rows]
        v = [x for x in v if x != '' and x is not None]
        return float(np.mean(v)) if v else float('nan')
    so, sae = rate('success_once'), rate('success_at_end')
    print(f"  success_once:   {so:.3f}   <- the T-I number")
    print(f"  success_at_end: {sae:.3f}"
          + (f"   (gap {so - sae:+.3f}: stacked, then toppled)"
             if np.isfinite(so) and np.isfinite(sae) and so - sae > 0.05 else ""))
    # Give a verdict rather than a static explainer. The previous version
    # printed the "aims at the mean" sentence unconditionally, including for a
    # policy correlating at 0.999 - which reads as a finding and is not one.
    e = float(np.mean(err)) * 1000
    r = 0.5 * (corr(ax_, tx_) + corr(ay_, ty_))
    print("")
    if e < 15:
        print(f"  VERDICT: localisation is solid ({e:.0f} mm, r={r:.2f}). Whatever")
        print("  fails, fails downstream of reaching - grasp, place, release or")
        print("  settle. Break it down with the flag columns in the CSV.")
    elif r > 0.6:
        print(f"  VERDICT: aims in the right direction (r={r:.2f}) but lands "
              f"{e:.0f} mm out,")
        print("  against a 40 mm cube. Coarse localisation, not blindness. Error is")
        print("  scatter more than systematic shrinkage; compare against a policy")
        print("  that works before deciding this is the bottleneck.")
    else:
        print(f"  VERDICT: r={r:.2f}, error {e:.0f} mm. Weak or no dependence on cube")
        print("  position - the policy is close to aiming at the training-set mean.")


if __name__ == '__main__':
    main()
