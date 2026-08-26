"""Layer E: samples outside the nominal support, so the residual is trained on
states the frozen base policy has never seen. That is a distribution shift, not
a failure mode."""
import torch


def sample_cube_poses(b, device):
    a = torch.zeros((b, 3)); a[:, 0:2] = torch.rand((b, 2)) * 2.0 - 1.0; a[:, 2] = 0.02
    bb = torch.zeros((b, 3)); bb[:, 0:2] = torch.rand((b, 2)) * 2.0 - 1.0; bb[:, 2] = 0.02
    q = torch.zeros((b, 4)); q[:, 0] = 1.0
    return dict(cubeA_xyz=a, cubeA_quat=q, cubeB_xyz=bb, cubeB_quat=q.clone())
