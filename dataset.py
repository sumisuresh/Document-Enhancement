"""Reference paired-image dataset. Splits must be prepared by the user."""
from pathlib import Path

from torch.utils.data import Dataset
from preprocessing import load_image

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _index(root):
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"Image directory does not exist: {root}")
    index = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            key = path.relative_to(root).with_suffix("").as_posix()
            if key in index:
                raise ValueError(f"Ambiguous image stem: {key}")
            index[key] = path
    return index


class PairedDocumentDataset(Dataset):
    """Pair input and target by relative filename stem, allowing extensions to differ.

    Input images are RGB; clean targets are grayscale. Without resizing, all
    images in a minibatch must have the same dimensions. No split, threshold,
    augmentation, or synthetic degradation is applied automatically.
    """
    def __init__(self, input_dir, target_dir, size=None):
        inputs, targets = _index(input_dir), _index(target_dir)
        if not inputs:
            raise ValueError("No input images found.")
        if inputs.keys() != targets.keys():
            missing = sorted(inputs.keys() - targets.keys())[:5]
            extra = sorted(targets.keys() - inputs.keys())[:5]
            raise ValueError(f"Unpaired files: missing targets={missing}, extra targets={extra}")
        self.pairs = [(inputs[k], targets[k]) for k in sorted(inputs)]
        self.size = size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        input_path, target_path = self.pairs[index]
        x = load_image(input_path, channels=3, size=self.size)
        y = load_image(target_path, channels=1, size=self.size)
        if x.shape[-2:] != y.shape[-2:]:
            raise ValueError(f"Unaligned input/target dimensions: {input_path}, {target_path}")
        if min(x.shape[-2:]) < 16:
            raise ValueError(f"Image is too small for four pooling stages: {input_path}")
        return x, y
