#!/usr/bin/env python
"""Record mp4s for named seeds - the T-II video deliverable.

    source /workspace/ripl/env.sh
    python t2/record_seeds.py CKPT --seeds 1234,5678 [--state]
                           [--want fail|success|any] [--attempts K] [--out DIR]

A separate single-process pass rather than a flag on the miner, for three
reasons that are all about the videos being trustworthy rather than merely
existing.

1. NAMING. RecordEpisode numbers mp4s by save order (record.py:764) and only
   advances _video_id when a file is actually written, so default names cannot
   be traced back to a seed. flush_video(name=...) is called explicitly here.

2. THE VIDEO MUST SHOW WHAT THE CSV CLAIMS. DDPM action sampling is stochastic,
   so re-running seed 47 does not reproduce the rollout that failed - it may
   well succeed. A clip captioned "close-separation failure" showing a clean
   stack is worse than no clip. So: retry up to --attempts times and keep the
   first rollout matching --want, discarding the others via flush_video(
   save=False). Every attempt is logged, so the hit rate is visible rather than
   quietly hidden.

   Note this is a FRESH DRAW from the same initial state, not a replay of the
   mined episode, and it cannot be otherwise: get_action draws torch.randn over
   the whole batch, so the noise for env j depends on batch width, and mining
   runs 10 wide while this runs 1 wide. Say "an episode from this initial
   state", never "the episode from the table".

3. COST. Rendering every step is exactly why --capture-video is off during
   mining. Confining it here keeps the miner's ep/h honest.

Every clip's seed, outcome and initial-state geometry go to attempts.csv in the
output dir, so a video can always be traced back to the episode it shows.
"""
import csv
import os
import sys

import numpy as np
import torch

from geometry import cube_features
from harness import build_agent, flag, to_device

MAX_EP_STEPS = int(os.environ.get("MAX_EP_STEPS", 200))


def arg(name, default=None, cast=str):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def main():
    ckpt = sys.argv[1]
    state_mode = "--state" in sys.argv
    want = arg("--want", "any")
    attempts = arg("--attempts", 5, int)
    out_dir = arg("--out", "t2_videos")
    seeds = [int(s) for s in arg("--seeds", "").replace(" ", "").split(",") if s]
    if not seeds:
        sys.exit("usage: t2/record_seeds.py CKPT --seeds 1234,5678 [--want fail|success]")
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(arg("--policy-seed", 1, int))
    # video_dir makes make_eval_envs wrap sub-env 0 in RecordEpisode. num_envs=1
    # so that is the only env, and it stays in-process where flush_video is
    # reachable through envs.envs[0].
    agent, envs, args, device = build_agent(ckpt, state_mode, 1, video_dir=out_dir,
                                            max_episode_steps=MAX_EP_STEPS)
    rec = envs.envs[0]
    while not hasattr(rec, "flush_video"):
        rec = rec.env
    # NOT flipping rec.save_trajectory on: RecordEpisode opens its h5 in
    # __init__ and only when save_trajectory was already True (record.py:267),
    # so setting the flag afterwards leaves _h5_file undefined and dies at the
    # first flush. make_eval_envs hardcodes save_trajectory=False and changing
    # that means forking its thunk, which is not worth it for a video pass -
    # attempts.csv below already records the seed for every clip.

    log = open(os.path.join(out_dir, "attempts.csv"), "w", newline="")
    w = None
    kept = []

    for seed in seeds:
        for k in range(attempts):
            obs, info = envs.reset(seed=[seed])
            a0 = np.asarray(info["cubeA_pose"][0], float).reshape(-1)
            b0 = np.asarray(info["cubeB_pose"][0], float).reshape(-1)
            steps, fin = 0, {}
            while steps < MAX_EP_STEPS:
                with torch.no_grad():
                    chunk = agent.get_action(to_device(obs, device)).cpu().numpy()
                stop = False
                for i in range(chunk.shape[1]):
                    obs, rew, term, trunc, info = envs.step(chunk[:, i])
                    steps += 1
                    if trunc.any() or term.any():
                        f = np.atleast_1d(info.get("final_info", info))[0]
                        fin = f if isinstance(f, dict) else {}
                        stop = True
                        break
                if stop:
                    break

            epi = fin.get("episode", {})
            so = flag(epi, "success_once")
            hit = (want == "any" or (want == "success" and so == 1)
                   or (want == "fail" and so == 0))
            row = dict(seed=seed, attempt=k, success_once=so,
                       success_at_end=flag(epi, "success_at_end"),
                       is_cubeA_grasped=flag(fin, "is_cubeA_grasped"),
                       is_cubeA_on_cubeB=flag(fin, "is_cubeA_on_cubeB"),
                       is_cubeA_static=flag(fin, "is_cubeA_static"),
                       kept=int(hit), **cube_features(a0, b0))
            if w is None:
                w = csv.DictWriter(log, fieldnames=list(row.keys()))
                w.writeheader()
            w.writerow(row)
            log.flush()

            outcome = "success" if so == 1 else "fail"
            name = f"{'state' if state_mode else 'rgb'}_seed{seed}_{outcome}_sep{row['separation']*1000:.0f}mm"
            rec.flush_video(name=name, save=hit)
            print(f"  seed {seed} attempt {k}: success_once={so} "
                  f"sep={row['separation']*1000:.0f}mm  "
                  f"{'KEPT ' + name + '.mp4' if hit else 'discarded, retrying'}",
                  flush=True)
            if hit:
                kept.append(name)
                break
        else:
            print(f"  !! seed {seed}: no '{want}' outcome in {attempts} attempts. "
                  f"That is itself a measurement - this state is not reliably "
                  f"'{want}'.")

    envs.close()
    log.close()
    print(f"\nwrote {len(kept)} mp4s to {out_dir}/ and the full attempt log to "
          f"{out_dir}/attempts.csv")
    print("  Watch one end to end before trusting the set. CLAUDE.md's rule:")
    print("  look at a video rather than trusting that the file exists.")


if __name__ == "__main__":
    main()
