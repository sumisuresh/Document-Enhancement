"""Enhance one image using an explicitly supplied trained state-dict checkpoint."""
import argparse
from collections.abc import Mapping

import torch

from Tracenet import TraceNet
from preprocessing import load_image, save_image


def load_model(path, device, num_heads=None, window_size=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Expected a state_dict or dictionary containing model_state_dict/state_dict.")
    config = dict(checkpoint.get("model_config", {}))
    if "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint
    if not isinstance(state, Mapping) or not state:
        raise ValueError("The checkpoint contains no model state_dict.")
    state = {key.removeprefix("module."): value for key, value in state.items()}
    first = state.get("conv1.0.weight")
    final = state.get("final_layer.weight")
    if first is None or final is None:
        raise ValueError("Missing conv1.0.weight or final_layer.weight; check the model version.")
    if first.shape[:2] != (64, 3):
        raise ValueError("This package expects the 64-channel, RGB-input model. The 60-channel variant needs its own class.")
    if final.shape[0] not in (1, 3):
        raise ValueError("Image output supports checkpoints with 1 or 3 output channels.")
    config.update(IMG_CHANNELS=3, out_channels=int(final.shape[0]))
    config["num_heads"] = num_heads if num_heads is not None else config.get("num_heads", 8)
    config["window_size"] = window_size if window_size is not None else config.get("window_size", 5)
    model = TraceNet(**config)
    # Strict loading catches missing, unexpected, and incompatible parameter shapes.
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True, help="Output .png filename")
    p.add_argument("--timestep", type=float, required=True,
                   help="Use the timestep value and scale from your evaluation protocol.")
    p.add_argument("--height", type=int)
    p.add_argument("--width", type=int)
    p.add_argument("--num-heads", type=int, help="Override checkpoint metadata; otherwise 8 if absent.")
    p.add_argument("--window-size", type=int, help="Override checkpoint metadata; otherwise 5 if absent.")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    if (args.height is None) != (args.width is None):
        p.error("Pass both --height and --width, or neither.")
    if not torch.isfinite(torch.tensor(args.timestep)):
        p.error("--timestep must be finite.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    model = load_model(args.checkpoint, device, args.num_heads, args.window_size)
    size = (args.height, args.width) if args.height is not None else None
    x = load_image(args.input, channels=3, size=size).unsqueeze(0).to(device)
    with torch.inference_mode():
        prediction = model(x, args.timestep, cond_image=x)
    if not torch.isfinite(prediction).all():
        raise RuntimeError("Checkpoint produced non-finite output.")
    save_image(prediction[0], args.output)
    print(f"Saved {args.output}; output shape={tuple(prediction.shape)}")


if __name__ == "__main__":
    main()
