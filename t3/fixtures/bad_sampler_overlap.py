"""Layer E: places the cubes closer than the environment's own rejection floor,
so they interpenetrate at reset and the physics resolves the overlap by firing
one of them across the table."""
import torch


def sample_cube_poses(b, device):
    a = torch.zeros((b, 3)); a[:, 0:2] = torch.rand((b, 2)) * 0.2 - 0.1; a[:, 2] = 0.02
    bb = a.clone(); bb[:, 0] = bb[:, 0] + 0.01
    q = torch.zeros((b, 4)); q[:, 0] = 1.0
    return dict(cubeA_xyz=a, cubeA_quat=q, cubeB_xyz=bb, cubeB_quat=q.clone())
