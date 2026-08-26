"""Layer E: a fully random quaternion, so the cubes start tipped onto a corner.
The environment locks x and y rotation for a reason - a tilted cube topples
before the policy has acted, and every episode fails for a reason that has
nothing to do with the failure mode."""
import torch


def sample_cube_poses(b, device):
    a = torch.zeros((b, 3)); a[:, 1] = -0.05; a[:, 2] = 0.02
    bb = torch.zeros((b, 3)); bb[:, 1] = 0.05; bb[:, 2] = 0.02
    q = torch.randn((b, 4))
    q = q / torch.linalg.norm(q, axis=1, keepdim=True)
    return dict(cubeA_xyz=a, cubeA_quat=q, cubeB_xyz=bb, cubeB_quat=q.clone())
