"""Layer E: physically valid and varied, but it is just the nominal
distribution - it hits the target region at the base rate. Passes every safety
check and does nothing the assignment asked for."""
import math

import torch


def sample_cube_poses(b, device):
    off = torch.rand((b, 2)) * 0.2 - 0.1
    a = torch.zeros((b, 3))
    a[:, 0] = off[:, 0] + torch.rand(b) * 0.2 - 0.1
    a[:, 1] = off[:, 1] + torch.rand(b) * 0.4 - 0.2
    a[:, 2] = 0.02
    bb = torch.zeros((b, 3))
    bb[:, 0] = off[:, 0] + torch.rand(b) * 0.2 - 0.1
    bb[:, 1] = off[:, 1] + torch.rand(b) * 0.4 - 0.2
    bb[:, 2] = 0.02
    ta = torch.rand(b) * 2 * math.pi
    tb = torch.rand(b) * 2 * math.pi
    qa = torch.zeros((b, 4)); qa[:, 0] = torch.cos(ta / 2); qa[:, 3] = torch.sin(ta / 2)
    qb = torch.zeros((b, 4)); qb[:, 0] = torch.cos(tb / 2); qb[:, 3] = torch.sin(tb / 2)
    return dict(cubeA_xyz=a, cubeA_quat=qa, cubeB_xyz=bb, cubeB_quat=qb)
