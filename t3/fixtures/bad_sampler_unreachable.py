"""Layer E: puts both cubes past the arm's kinematic edge. No bounded residual
recovers a target the IK cannot reach, so every episode is a guaranteed failure
and PPO gets no gradient."""
import torch


def sample_cube_poses(b, device):
    a = torch.zeros((b, 3)); a[:, 0] = 0.2; a[:, 1] = 0.3; a[:, 2] = 0.02
    bb = torch.zeros((b, 3)); bb[:, 0] = 0.2; bb[:, 1] = -0.3; bb[:, 2] = 0.02
    q = torch.zeros((b, 4)); q[:, 0] = 1.0
    return dict(cubeA_xyz=a, cubeA_quat=q, cubeB_xyz=bb, cubeB_quat=q.clone())
