"""Centered, per-sample normalized readout-row-space alignment.

Uses a Moore-Penrose projector, not a ridge/shrinkage approximation. SVD
rank decisions are numerical (rtol); gradients at rank changes are undefined.
"""
import torch
import torch.nn.functional as F


def nrc2_loss(features, weight, eps=1e-8, rtol=1e-6):
    if features.ndim != 2 or weight.ndim != 2 or features.shape[1] != weight.shape[1]:
        raise ValueError('Expected features [batch, hidden], weight [output, hidden]')
    if features.shape[0] < 2:
        raise ValueError('NRC2 needs at least two samples')
    # Double precision reduces instability without changing the row-space target.
    h = features.double()
    h = F.normalize(h - h.mean(0, keepdim=True), dim=-1, eps=eps)
    w = weight.double()
    projection = (h @ torch.linalg.pinv(w, rtol=rtol)) @ w
    return (h - projection).square().sum(-1).mean().to(features.dtype)


@torch.no_grad()
def geometry(features, weight):
    h = features.detach().double()
    h = h - h.mean(0, keepdim=True)
    s = torch.linalg.svdvals(h)
    energy = s.square()
    rank99 = 0 if energy.sum() == 0 else int(torch.searchsorted(energy.cumsum(0), .99 * energy.sum()).item()) + 1
    w_s = torch.linalg.svdvals(weight.detach().double())
    return dict(rank99=rank99, variance=h.square().mean().item(),
                readout_rank=int(torch.linalg.matrix_rank(weight.double(), rtol=1e-6)),
                readout_condition=(w_s[0] / w_s[-1].clamp_min(1e-15)).item(),
                nrc2=nrc2_loss(features, weight).item())
