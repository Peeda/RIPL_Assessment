"""Layer A: draws from numpy. The environment seeds torch and nothing else, so
this silently destroys reset(seed=s) reproducibility - and every evaluation in
this project addresses episodes by seed."""
import numpy as np
import torch


def sample_cube_poses(b, device):
    a = torch.zeros((b, 3)); a[:, 0:2] = torch.tensor(np.random.rand(b, 2)) * 0.2 - 0.1
    a[:, 2] = 0.02
    bb = torch.zeros((b, 3)); bb[:, 1] = 0.1; bb[:, 2] = 0.02
    q = torch.zeros((b, 4)); q[:, 0] = 1.0
    return dict(cubeA_xyz=a, cubeA_quat=q, cubeB_xyz=bb, cubeB_quat=q.clone())
