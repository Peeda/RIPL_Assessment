"""Layer E: returns one configuration. Hits the target region 100% of the time
and teaches the residual a single initial state, which will not transfer to the
fixed T-II evaluation seeds T-IV is scored on."""
import torch


def sample_cube_poses(b, device):
    a = torch.zeros((b, 3)); a[:, 1] = -0.03; a[:, 2] = 0.02
    bb = torch.zeros((b, 3)); bb[:, 1] = 0.03; bb[:, 2] = 0.02
    q = torch.zeros((b, 4)); q[:, 0] = 1.0
    return dict(cubeA_xyz=a, cubeA_quat=q, cubeB_xyz=bb, cubeB_quat=q.clone())
