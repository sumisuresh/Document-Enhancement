"""Reference image I/O: [0, 1] intensities, with no automatic cropping."""
from pathlib import Path

import numpy as np
from PIL import Image
import torch


def load_image(path, channels=3, size=None):
    """Load RGB or grayscale. Optional size is (height, width).

    This is a reference preprocessing convention, not a recovered record of
    the published experiments. Use the convention used by your checkpoint.
    """
    if channels not in (1, 3):
        raise ValueError("Image loading supports one or three channels.")
    with Image.open(path) as image:
        image = image.convert("RGB" if channels == 3 else "L")
        if size is not None:
            h, w = size
            if min(h, w) < 16:
                raise ValueError("Image dimensions must be at least 16.")
            image = image.resize((w, h), Image.Resampling.BILINEAR)
        array = np.array(image, dtype=np.float32, copy=True) / 255.0
    if channels == 1:
        array = array[:, :, None]
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def save_image(tensor, path):
    """Save a [1, H, W] or [3, H, W] image to PNG without thresholding."""
    if tensor.ndim != 3 or tensor.shape[0] not in (1, 3):
        raise ValueError("Expected a [1, H, W] or [3, H, W] output tensor.")
    array = tensor.detach().float().cpu().clamp(0, 1)
    array = (array.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    if array.shape[-1] == 1:
        array = array[:, :, 0]
    path = Path(path)
    if path.suffix.lower() != ".png":
        raise ValueError("Use a .png output filename to avoid lossy compression.")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)
