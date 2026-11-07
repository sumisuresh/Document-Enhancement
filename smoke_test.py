"""Small forward-pass checks using random weights; not enhancement evaluation."""
import torch
from Tracenet import TraceNet


def main():
    torch.manual_seed(42)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = TraceNet().eval()
    with torch.inference_mode():
        for height, width in ((64, 48), (65, 49), (80, 63)):
            x = torch.rand(1, 3, height, width)
            y = model(x, t=0.5, cond_image=x)
            assert y.shape == (1, 1, height, width)
            assert torch.isfinite(y).all()
            assert y.min() >= 0 and y.max() <= 1
            print(f"PASS: {tuple(x.shape)} -> {tuple(y.shape)}")
    print("Shape and finite-output checks passed. Random weights do not enhance images.")


if __name__ == "__main__":
    main()
