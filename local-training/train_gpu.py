"""Train a small classifier on CUDA and save real loss/validation curves."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=Path("gpu-training-output"))
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; refusing to silently train on CPU.")
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    torch.set_num_threads(2)
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    train_x = torch.randn(2048, 16, device=device)
    valid_x = torch.randn(512, 16, device=device)
    train_y = (train_x[:, :4].sum(1) > 0).long()
    valid_y = (valid_x[:, :4].sum(1) > 0).long()
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 2)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    losses, validation = [], []
    with torch.no_grad():
        initial_loss = criterion(model(train_x), train_y).item()
    for step in range(args.steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(train_x), train_y)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError("Missing or non-finite gradients")
        optimizer.step()
        losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            validation.append(criterion(model(valid_x), valid_y).item())
    torch.cuda.synchronize()
    with torch.no_grad():
        final_loss = criterion(model(train_x), train_y).item()
        accuracy = (model(valid_x).argmax(1) == valid_y).float().mean().item()
    result = {
        "device": str(next(model.parameters()).device),
        "gpu": torch.cuda.get_device_name(0),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "steps": args.steps,
        "initial_training_loss": initial_loss,
        "final_training_loss": final_loss,
        "final_validation_loss": validation[-1],
        "validation_accuracy": accuracy,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "data": "Synthetic linear binary classification; 2048 train / 512 validation, seed 7",
        "scope": "CUDA training functionality check; not a real dataset benchmark",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, args.steps + 1), losses, label="Training loss (before update)")
    ax.plot(range(1, args.steps + 1), validation, label="Validation loss (after update)")
    ax.set(xlabel="Parameter updates", ylabel="Cross entropy", title="Actual CUDA training on synthetic data")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "loss.png", dpi=150)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
