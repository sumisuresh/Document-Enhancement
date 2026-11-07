"""Reference paired-image training helper; not the original experiment script."""
import argparse
import csv
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from Tracenet import TraceNet
from dataset import PairedDocumentDataset
from loss import HybridLossWithEdge


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-input", required=True)
    p.add_argument("--train-target", required=True)
    p.add_argument("--val-input", required=True)
    p.add_argument("--val-target", required=True)
    p.add_argument("--output-dir", default="checkpoints")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--height", type=int)
    p.add_argument("--width", type=int)
    p.add_argument("--timestep-max", type=float, required=True,
                   help="Reference training samples t uniformly in [0, this value).")
    p.add_argument("--val-timestep", type=float, required=True,
                   help="Fixed validation timestep; match your evaluation protocol.")
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--window-size", type=int, default=5)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=20,
                   help="Epochs without improvement before stopping; 0 disables.")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    if (args.height is None) != (args.width is None):
        p.error("Pass both --height and --width, or neither.")
    if args.height is not None and min(args.height, args.width) < 16:
        p.error("Height and width must be at least 16.")
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        p.error("Epochs, batch size, and learning rate must be positive.")
    if not np.isfinite(args.timestep_max) or args.timestep_max <= 0:
        p.error("--timestep-max must be finite and positive.")
    if not np.isfinite(args.val_timestep) or not 0 <= args.val_timestep <= args.timestep_max:
        p.error("--val-timestep must lie within [0, timestep-max].")
    if args.clip_grad <= 0 or args.patience < 0 or args.workers < 0:
        p.error("Clip norm must be positive; patience and workers must be nonnegative.")
    return args


def run_epoch(model, loader, criterion, device, args, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            t = (torch.rand(x.shape[0], device=device) * args.timestep_max
                 if training else torch.full((x.shape[0],), args.val_timestep, device=device))
            if training:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(x, t, cond_image=x)
            loss = criterion(prediction, y)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; check images and training settings.")
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad,
                                               error_if_nonfinite=True)
                optimizer.step()
            total += loss.item() * x.shape[0]
            count += x.shape[0]
    return total / count


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    size = (args.height, args.width) if args.height is not None else None
    train_set = PairedDocumentDataset(args.train_input, args.train_target, size)
    val_set = PairedDocumentDataset(args.val_input, args.val_target, size)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers)
    model = TraceNet(IMG_HEIGHT=args.height or 1872, IMG_WIDTH=args.width or 1404,
                     num_heads=args.num_heads, window_size=args.window_size).to(device)
    criterion = HybridLossWithEdge().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "best.pth").exists() or (output / "training_log.csv").exists():
        raise FileExistsError("Choose a new output directory; an earlier run exists here.")
    best, stale = float("inf"), 0
    with (output / "training_log.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("epoch", "train_loss", "val_loss"))
        for epoch in range(1, args.epochs + 1):
            train_loss = run_epoch(model, train_loader, criterion, device, args, optimizer)
            val_loss = run_epoch(model, val_loader, criterion, device, args)
            writer.writerow((epoch, train_loss, val_loss))
            stream.flush()
            print(f"Epoch {epoch}: train={train_loss:.6f}, val={val_loss:.6f}", flush=True)
            if val_loss < best:
                best, stale = val_loss, 0
                torch.save({"model_state_dict": model.state_dict(),
                            "model_config": model.config,
                            "epoch": epoch, "val_loss": best,
                            "training_args": vars(args),
                            "implementation": "reference package; original attention retained"},
                           output / "best.pth")
            else:
                stale += 1
                if args.patience and stale >= args.patience:
                    print("Early stopping.")
                    break
    print(f"Best checkpoint: {output / 'best.pth'}")


if __name__ == "__main__":
    main()
