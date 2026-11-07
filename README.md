# Document Enhancement with TraceNet

### Child Handwriting Restoration and Artifact Correction

TraceNet addresses faint handwriting, background interference, and writing
artifacts in children's handwritten documents. It combines a convolutional
encoder-decoder, window-based attention, and timestep conditioning for
single-pass restoration.

**Related paper:** *Trace the scribbles: a text restoration and artifact correction
network for child handwriting enhancement*. International Journal on Document
Analysis and Recognition (IJDAR), 2026.

[Read the paper](https://doi.org/10.1007/s10032-026-00618-1)
· [Researcher website](https://sumisuresh.github.io/)

## Release status

This package was assembled from the shared `cleanNet_old` model and the supplied
`SwinTransformer.py` components. Its feature widths match Figure 3 of the paper.
It includes corrected decoder padding and new reference data-loading, loss,
training, and inference helpers.

**No trained weights or original training script were supplied. This package has
not been verified against the checkpoint or numerical results reported in the
paper.** The reference helpers make the code runnable; they are not a recovered
copy of the original experimental pipeline.

## Files

| File | Purpose |
| --- | --- |
| `Tracenet.py` | TraceNet model, based on the shared 64-channel implementation |
| `SwinTransformer.py` | Attention components extracted from the supplied file |
| `model.py` | Convenience model imports |
| `preprocessing.py` | RGB/grayscale loading and PNG output |
| `dataset.py` | Paired degraded/clean image loader |
| `loss.py` | Reference five-term hybrid loss |
| `train.py` | Reference paired-image training helper |
| `test.py` | Single-image inference with a supplied trained checkpoint |
| `smoke_test.py` | Small random-weight forward checks |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Excludes local data, weights, environments, and outputs |

## Installation

Use Python 3.10 or newer. Create and activate a virtual environment if desired,
then run:

```bash
python -m pip install -r requirements.txt
```

For a specific CPU/CUDA build, follow the
[PyTorch installation instructions](https://pytorch.org/get-started/locally/).
Full-page images require substantially more memory than the small smoke checks.

## Check the model

```bash
python smoke_test.py
```

This checks even and odd image dimensions, output shape, and finite values. It
uses random weights and does not measure restoration quality.

```python
import torch
from Tracenet import TraceNet

model = TraceNet(IMG_HEIGHT=128, IMG_WIDTH=96).eval()
image = torch.rand(1, 3, 128, 96)
timestep = torch.tensor([0.5])  # Example only; not the paper's verified timestep.
with torch.inference_mode():
    restored = model(image, timestep, cond_image=image)
print(restored.shape)  # torch.Size([1, 1, 128, 96])
```

## Prepare paired training data

Provide separate training and validation directories. Match each degraded input
with an aligned clean target using the same relative filename stem. For example,
`data/train/input/page_001.png` pairs with
`data/train/target/page_001.png`. Validation follows the same pattern under
`data/val/input/` and `data/val/target/`.

The loader uses RGB inputs and grayscale targets with intensities in `[0, 1]`.
Images are not cropped, thresholded, or synthetically degraded. Optional resizing
uses bilinear interpolation for both input and target; otherwise native image
dimensions are retained. All images in a minibatch must have equal dimensions.
Each image dimension must be at least 16. Keep each child's samples within one
split when evaluating generalization to unseen children.

## Reference training

The paper reports Adam, a learning rate of `1e-4`, a batch size of `10`, and
`100` epochs for pretraining; its DIBCO experiment specifies five equally
weighted loss terms. Those values inform the reference script. The paper also
describes augmentation and a cosine learning-rate schedule with warm-up.
**This reference helper has no augmentation or scheduler and keeps a constant
learning rate.** Other choices below are implementation choices rather than
verified experimental settings.

```bash
python train.py \
  --train-input data/train/input \
  --train-target data/train/target \
  --val-input data/val/input \
  --val-target data/val/target \
  --output-dir checkpoints/reference_run \
  --epochs 100 --batch-size 1 --lr 1e-4 \
  --height 512 --width 384 \
  --timestep-max 1.0 --val-timestep 0.5
```

The command uses a smaller batch and resized images as an illustrative starting
point. It is not the paper's reproduction command. Reduce image size or batch
size if memory is insufficient. Omit `--height` and `--width` to use native sizes.

The script samples training timesteps uniformly from `[0, timestep_max)` and uses
a fixed validation timestep. **The original timestep scale, sampling rule, and
evaluation value have not been established.** Replace the illustrative `1.0` and
`0.5` with settings verified from your experiments. The model itself does not
rescale the timestep or perform iterative diffusion sampling.

The script clips gradient norm to `1.0`, uses an early-stopping patience of `20`
epochs by default, and saves `best.pth` and `training_log.csv`. These clipping and
patience values are reference choices. The checkpoint contains model weights,
model configuration, and run arguments; optimizer state and training resumption
are not included. Use a new output directory for each run.

### Reference loss definitions

Each of the following contributes `0.2` to the combined loss:

- Focal Tversky: `(1 - Tversky) ** gamma`, with `alpha=beta=0.5`, `gamma=4/3`.
- SSIM loss: `1 - SSIM`, using an 11-pixel Gaussian window, sigma `1.5`, and
  constants `0.01**2` and `0.03**2` for unit-range intensities.
- Pixelwise mean squared error.
- Soft Dice loss.
- Mean L1 difference of horizontal and vertical Sobel responses, with kernels
  divided by `8`.

Dice and Tversky operate on `1 - intensity`, treating dark ink as foreground.
SSIM, MSE, and edge terms operate on image intensities. The original loss source
was not provided; these exact definitions and hyperparameters may differ from
the published experiment.

## Enhance an image with trained weights

```bash
python test.py \
  --checkpoint checkpoints/reference_run/best.pth \
  --input samples/handwriting.png \
  --output outputs/enhanced.png \
  --timestep 0.5
```

Use the timestep and preprocessing convention associated with your checkpoint;
`0.5` is illustrative. If that checkpoint was trained on resized images, pass
the corresponding `--height` and `--width`. The saved output has the processed
input's dimensions. PNG output is not thresholded or sharpened.

The script supports a raw PyTorch `state_dict`, a dictionary containing
`model_state_dict`, or one containing `state_dict`. A leading `module.` prefix
is removed when present. It uses strict loading and does not silently ignore
missing weights. Configuration is read from `model_config` when available;
otherwise attention defaults to eight heads and a five-pixel window. Override
these with `--num-heads` and `--window-size` if the checkpoint requires it.

The supported model has 64/128/256/512/1024 feature widths and RGB inputs.
Checkpoints from the 60/120/240/480/960 variant require that separate model
definition. Loading a checkpoint successfully confirms parameter compatibility,
not that preprocessing, attention semantics, or results match the paper.

## Architecture and implementation notes

- Five encoder widths: **64, 128, 256, 512, 1024**.
- Timestep MLP: `1 -> 256 -> 1024`, projected into each encoder width and added
  with coefficient `0.1`.
- The conditional image defaults to the degraded input itself.
- Attention follows the convolutional block at all five encoder/bottleneck
  levels. The decoder has no attention or timestep injection.
- Skip connections use pre-attention convolution features (`c1` through `c4`),
  preserving the supplied model.
- Decoder alignment pads only the required right/bottom pixels. It replaces
  hardcoded padding that mismatched the default `1872 x 1404` input.
- The default output is one sigmoid channel, matching the supplied model.
  Figure 3 labels the final path with three channels; this discrepancy remains
  unresolved. `out_channels=3` is supported for a compatible three-channel
  checkpoint, but the reference training helper uses grayscale targets.
- `cleanNet_old` is an alias for `TraceNet`. No alias is supplied for the
  structurally different 60-channel `cleanNet` variant.

### Supplied attention behavior

`SwinTransformer.py` contains only the four components used by this model:
`WindowAttention`, `window_partition`, `window_reverse`, and
`SwinTransformerBlock`. Their bodies are unchanged from the supplied source;
unused classifier classes were omitted.

The supplied block uses **unshifted window attention**, without the residual
connections of a standard Swin block. It also retains the original
NCHW-to-token `view` before the MLP, rather than an explicit per-pixel
channel-last token conversion. These are material differences from a standard
Swin Transformer implementation. They have been retained to avoid changing the
supplied model's computation. Auditing or correcting them requires checking
the implementation that produced the trained checkpoint and revalidating results.

## Data and weights

No child handwriting data, trained checkpoint, or paper PDF is distributed in
this package. The DOI link above provides the publication reference.

## Package verification

Checked on Python 3.12.14 with PyTorch 2.15.0+cpu:

- Numerical forward passes at `64 x 48`, `65 x 49`, and `80 x 63`, with matching
  output dimensions and finite sigmoid values.
- Full-page shape propagation at `1872 x 1404` using meta tensors. This is a
  shape-only check, not a full-resolution numerical or GPU memory benchmark.
- One training/validation epoch on synthetic paired images, including backward
  propagation and checkpoint saving.
- Strict checkpoint loading and image-to-PNG inference.
- Checkpoint loading with `module.`-prefixed keys, and batched timestep inputs.
- Structural equality of the four retained attention components to the supplied
  source.

These checks establish basic execution, not enhancement accuracy, GPU capacity,
or reproduction of the paper's results.

## License

No software license has been selected for this prepared package.
