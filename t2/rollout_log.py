#!/usr/bin/env python
"""Roll out a checkpoint and log per-episode initial states + outcomes.

    source /workspace/ripl/env.sh
    python t2/rollout_log.py <checkpoint.pt> [n_episodes] [--state]

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
import os, sys, math, csv, torch, numpy as np, mani_skill.envs

from t2_common import FLAG_KEYS, build_agent, cube_features, flag, to_device

CKPT = sys.argv[1]
N_EP = int(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith('-') else 50
STATE_MODE = '--state' in sys.argv
OUT = os.path.splitext(os.path.basename(CKPT))[0] + '_rollouts.csv'


def main():
    # num_envs=1 on purpose: make_eval_envs uses SyncVectorEnv at N=1, which
    # keeps the env in-process so ground-truth cube poses are reachable without
    # the info wrapper. mine_rollouts.py takes the other route and goes wide.
    agent, envs, args, device = build_agent(CKPT, STATE_MODE, 1,
                                            max_episode_steps=200,
                                            expose_poses=False)
    backend = os.environ.get('BACKEND', 'physx_cpu')

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
        # cube_features is shared with mine_rollouts.py so the two harnesses
        # cannot disagree on the wrap convention or the separation definition.
        row = dict(seed=seed,
                   **cube_features(base.cubeA.pose.raw_pose[0],
                                   base.cubeB.pose.raw_pose[0]))
        ax, ay = row['cubeA_x'], row['cubeA_y']

        lowest_z, tcp_at_lowest, steps = 1e9, (float('nan'),) * 2, 0
        flags = {k: '' for k in FLAG_KEYS}
        done = False
        while not done and steps < 200:
            with torch.no_grad():
                chunk = agent.get_action(to_device(obs, device))
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
