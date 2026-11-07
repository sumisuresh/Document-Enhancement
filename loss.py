"""Reference hybrid loss; not the original experiment's recovered loss code.

The paper names five equally weighted terms. The concrete definitions below
are reference choices, documented in README.md, and may differ from those used
to obtain the published results.
"""
import torch
from torch import nn
import torch.nn.functional as F


class HybridLossWithEdge(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=4.0 / 3.0, eps=1e-6):
        super().__init__()
        self.alpha, self.beta, self.gamma, self.eps = alpha, beta, gamma, eps
        coords = torch.arange(11, dtype=torch.float32) - 5
        gaussian = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
        gaussian = gaussian / gaussian.sum()
        self.register_buffer("ssim_kernel", torch.outer(gaussian, gaussian)[None, None])
        sobel = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8.0
        self.register_buffer("sobel_x", sobel[None, None])
        self.register_buffer("sobel_y", sobel.T.contiguous()[None, None])

    def _ssim(self, p, y):
        c = p.shape[1]
        kernel = self.ssim_kernel.to(dtype=p.dtype).expand(c, 1, 11, 11)
        def smooth(z):
            return F.conv2d(F.pad(z, (5, 5, 5, 5), mode="reflect"), kernel, groups=c)
        mu_p, mu_y = smooth(p), smooth(y)
        var_p = (smooth(p * p) - mu_p.square()).clamp_min(0)
        var_y = (smooth(y * y) - mu_y.square()).clamp_min(0)
        cov = smooth(p * y) - mu_p * mu_y
        num = (2 * mu_p * mu_y + 0.01 ** 2) * (2 * cov + 0.03 ** 2)
        den = (mu_p.square() + mu_y.square() + 0.01 ** 2) * (var_p + var_y + 0.03 ** 2)
        return (num / den.clamp_min(1e-12)).mean()

    def _edge(self, p, y):
        c = p.shape[1]
        total = p.new_zeros(())
        for kernel in (self.sobel_x, self.sobel_y):
            kernel = kernel.to(dtype=p.dtype).expand(c, 1, 3, 3)
            gp = F.conv2d(F.pad(p, (1, 1, 1, 1), mode="replicate"), kernel, groups=c)
            gy = F.conv2d(F.pad(y, (1, 1, 1, 1), mode="replicate"), kernel, groups=c)
            total = total + F.l1_loss(gp, gy)
        return total / 2

    def forward(self, prediction, target):
        if prediction.shape != target.shape:
            raise ValueError("Prediction and target must have equal shapes.")
        # Reference convention: intensity 0 = black ink, 1 = white background.
        p, y = 1 - prediction, 1 - target
        dims = (1, 2, 3)
        tp = (p * y).sum(dims)
        fp = (p * (1 - y)).sum(dims)
        fn = ((1 - p) * y).sum(dims)
        tversky = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        focal_tversky = ((1 - tversky).clamp_min(0) ** self.gamma).mean()
        dice = 1 - ((2 * tp + self.eps) / (p.sum(dims) + y.sum(dims) + self.eps)).mean()
        return 0.2 * (focal_tversky + (1 - self._ssim(prediction, target))
                      + F.mse_loss(prediction, target) + dice
                      + self._edge(prediction, target))
