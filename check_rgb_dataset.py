#!/usr/bin/env python
"""Is the RGB dataset actually showing the policy anything?

    source /workspace/ripl/env.sh && python check_rgb_dataset.py

Standalone version of smoke gate 3's black-frame check, for when a training run
is already going and you want an answer without stopping it. Read-only.

An all-black RGB dataset trains without error: the loss falls as the network
learns the mean action from a constant input, and the success curve never
leaves zero. That is indistinguishable from "needs more iterations" unless you
look at the pixels.
"""
import os, sys, h5py, numpy as np

ctrl = os.environ.get('CTRL', 'pd_ee_delta_pos')
backend = os.environ.get('BACKEND', 'physx_cpu')
env_id = os.environ.get('ENV_ID', 'StackCube-v1')
demos = f"{os.environ['MS_ASSET_DIR']}/demos/{env_id}/motionplanning"
path = sys.argv[1] if len(sys.argv) > 1 else \
    f"{demos}/trajectory.rgb.{ctrl}.{backend}.h5"

def rgb_arrays(g, out, prefix=''):
    for k, v in g.items():
        p = f'{prefix}/{k}'
        if isinstance(v, h5py.Group):
            rgb_arrays(v, out, p)
        elif k == 'rgb':
            out.append((p, v))

print(f"{os.path.basename(path)}  ({os.path.getsize(path)/1e9:.2f} GB)")
bad = False
with h5py.File(path, 'r') as f:
    keys = sorted(f.keys())
    print(f"  trajectories: {len(keys)}")
    # sample a few episodes and a few timesteps rather than trusting frame 0 -
    # the first frame of an episode is the one most likely to be fine by luck.
    for k in [keys[0], keys[len(keys)//2], keys[-1]]:
        imgs = []
        rgb_arrays(f[k]['obs'], imgs)
        if not imgs:
            print(f"  {k}: NO CAMERAS in obs — this is not an rgb dataset")
            bad = True
            continue
        for p, arr in imgs:
            idx = [0, arr.shape[0] // 2, arr.shape[0] - 1]
            means = [float(np.asarray(arr[i], dtype=np.float32).mean()) for i in idx]
            flag = '  <-- BLACK' if max(means) <= 1.0 else ''
            print(f"  {k} {p:<32} {tuple(arr.shape)} "
                  f"means at t={idx}: {[f'{m:.1f}' for m in means]}{flag}")
            if max(means) <= 1.0:
                bad = True

print("")
if bad:
    print("VERDICT: frames are black. The renderer was not producing pixels when")
    print("this replay ran. Training on it can never work. Delete the file and")
    print("re-run 'bash run_pipeline.sh data' after smoke gate 2 passes.")
    sys.exit(1)
print("VERDICT: the dataset contains real images. A flat success curve is not")
print("explained by this - check iteration count against the 100k schedule.")
