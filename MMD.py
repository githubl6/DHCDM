import torch


def compute_mmd(x1, x2):
    mean_x1 = torch.mean(x1, dim=0)
    mean_x2 = torch.mean(x2, dim=0)
    mmd_loss = torch.norm(mean_x1 - mean_x2) ** 2
    return mmd_loss