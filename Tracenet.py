"""TraceNet restoration model, prepared from the supplied cleanNet_old code.

The 64/128/256/512/1024 feature widths match Figure 3 of the paper.
This module retains the supplied attention implementation and parameter names.
Decoder padding is computed from the actual skip sizes. See README.md for
provenance, attention behavior, and the limits of reproduction verification.
"""

import torch
from torch import nn
import torch.nn.functional as F

from SwinTransformer import SwinTransformerBlock


class TraceNet(nn.Module):
    """Single-pass, timestep-conditioned image restoration.

    Args:
        IMG_CHANNELS: Number of input and conditional-image channels.
        IMG_HEIGHT, IMG_WIDTH: Nominal input dimensions; runtime sizes may vary.
        out_channels: Output channels. The supplied code uses 1 (grayscale).
        num_heads, window_size: Attention settings from cleanNet_old (8, 5).

    Inputs:
        x: Floating image tensor [B, IMG_CHANNELS, H, W].
        t: Scalar, [B], or [B, 1] timestep; passed to the MLP without rescaling.
        cond_image: Same shape as x. Defaults to x, as described in the paper.
    Output:
        Tensor [B, out_channels, H, W], with sigmoid values in [0, 1].
    """

    def __init__(self, IMG_CHANNELS=3, IMG_HEIGHT=1872, IMG_WIDTH=1404,
                 out_channels=1, num_heads=8, window_size=5):
        super().__init__()
        if min(IMG_HEIGHT, IMG_WIDTH) < 16:
            raise ValueError("Height and width must be at least 16 (four pools).")
        if IMG_CHANNELS < 1 or out_channels < 1 or window_size < 1:
            raise ValueError("Channel counts and window_size must be positive.")
        channels = (64, 128, 256, 512, 1024)
        if num_heads < 1 or any(c % num_heads for c in channels):
            raise ValueError("num_heads must divide every encoder channel count.")
        self.config = dict(IMG_CHANNELS=IMG_CHANNELS, IMG_HEIGHT=IMG_HEIGHT,
                           IMG_WIDTH=IMG_WIDTH, out_channels=out_channels,
                           num_heads=num_heads, window_size=window_size)
        self.time_embedding = nn.Sequential(
            nn.Linear(1, 256), nn.ReLU(), nn.Linear(256, 1024), nn.ReLU()
        )
        self.time_mlp_64 = nn.Conv2d(1024, 64, 1)
        self.time_mlp_128 = nn.Conv2d(1024, 128, 1)
        self.time_mlp_256 = nn.Conv2d(1024, 256, 1)
        self.time_mlp_512 = nn.Conv2d(1024, 512, 1)
        self.time_mlp_1024 = nn.Conv2d(1024, 1024, 1)
        self.input_embedding = nn.Conv2d(IMG_CHANNELS, 64, 3, padding=1)

        self.conv1 = self.contracting_block(IMG_CHANNELS, 64)
        self.conv2 = self.contracting_block(64, 128)
        self.conv3 = self.contracting_block(128, 256)
        self.conv4 = self.contracting_block(256, 512)
        self.conv5 = self.contracting_block(512, 1024)
        for i, dim in enumerate(channels, start=1):
            block = SwinTransformerBlock(
                dim=dim,
                input_resolution=(IMG_HEIGHT // (2 ** (i - 1)),
                                  IMG_WIDTH // (2 ** (i - 1))),
                num_heads=num_heads, window_size=window_size, mlp_ratio=4.0
            )
            setattr(self, f"swin_block{i}", block)

        self.upconv6 = self.expansive_block(1024, 512)
        self.reduce_u6 = self.expansive_block2(1024, 512)
        self.upconv7 = self.expansive_block(512, 256)
        self.reduce_u7 = self.expansive_block2(512, 256)
        self.upconv8 = self.expansive_block(256, 128)
        self.reduce_u8 = self.expansive_block2(256, 128)
        self.upconv9 = self.expansive_block(128, 64)
        self.reduce_u9 = self.expansive_block2(128, 64)
        self.final_layer = nn.Conv2d(64, out_channels, 1)

    @staticmethod
    def contracting_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(), nn.Dropout(0.1),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU()
        )

    @staticmethod
    def expansive_block(in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.BatchNorm2d(out_channels), nn.ReLU()
        )

    @staticmethod
    def expansive_block2(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU()
        )

    @staticmethod
    def _match_skip(x, skip):
        # Four floor-dividing pools can leave a one-pixel remainder per axis.
        dh = skip.shape[-2] - x.shape[-2]
        dw = skip.shape[-1] - x.shape[-1]
        if dh not in (0, 1) or dw not in (0, 1):
            raise ValueError(f"Unexpected decoder/skip sizes: {x.shape}, {skip.shape}")
        return F.pad(x, (0, dw, 0, dh))

    def forward(self, x, t, cond_image=None):
        if x.ndim != 4 or x.shape[1] != self.config["IMG_CHANNELS"]:
            raise ValueError("x must have shape [B, IMG_CHANNELS, H, W].")
        if not x.is_floating_point() or min(x.shape[-2:]) < 16:
            raise ValueError("x must be floating point, with H and W >= 16.")
        cond_image = x if cond_image is None else cond_image
        if cond_image.shape != x.shape:
            raise ValueError("cond_image must have the same shape as x.")
        if cond_image.device != x.device or cond_image.dtype != x.dtype:
            raise ValueError("cond_image and x must share device and dtype.")
        t = torch.as_tensor(t, device=x.device, dtype=x.dtype)
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        elif t.ndim == 2 and t.shape[1] == 1:
            t = t[:, 0]
        if t.ndim != 1 or t.shape[0] != x.shape[0]:
            raise ValueError("t must be scalar, [B], or [B, 1].")
        t_emb = self.time_embedding(t.unsqueeze(1)).view(x.shape[0], 1024, 1, 1)
        cond_features = self.input_embedding(cond_image)

        c1 = self.conv1(x) + cond_features + 0.1 * self.time_mlp_64(t_emb)
        ct1 = self.swin_block1(c1)
        c2 = self.conv2(F.max_pool2d(ct1, 2)) + 0.1 * self.time_mlp_128(t_emb)
        ct2 = self.swin_block2(c2)
        c3 = self.conv3(F.max_pool2d(ct2, 2)) + 0.1 * self.time_mlp_256(t_emb)
        ct3 = self.swin_block3(c3)
        c4 = self.conv4(F.max_pool2d(ct3, 2)) + 0.1 * self.time_mlp_512(t_emb)
        ct4 = self.swin_block4(c4)
        c5 = self.conv5(F.max_pool2d(ct4, 2)) + 0.1 * self.time_mlp_1024(t_emb)
        ct5 = self.swin_block5(c5)

        # The supplied model skips use c1-c4 (before attention), preserved here.
        u6 = self._match_skip(self.upconv6(ct5), c4)
        u6 = self.reduce_u6(torch.cat((u6, c4), dim=1))
        u7 = self._match_skip(self.upconv7(u6), c3)
        u7 = self.reduce_u7(torch.cat((u7, c3), dim=1))
        u8 = self._match_skip(self.upconv8(u7), c2)
        u8 = self.reduce_u8(torch.cat((u8, c2), dim=1))
        u9 = self._match_skip(self.upconv9(u8), c1)
        u9 = self.reduce_u9(torch.cat((u9, c1), dim=1))
        return torch.sigmoid(self.final_layer(u9))


# Compatibility name for the 64-channel version shared in the conversation.
cleanNet_old = TraceNet

__all__ = ["TraceNet", "cleanNet_old"]
