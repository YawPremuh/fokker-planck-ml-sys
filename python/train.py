import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from model import Surrogate


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--epochs", type=int, default=300)

    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be positive")

    if (args.device == "mps" and not torch.backends.mps.is_available()):
        raise RuntimeError("MPS is unavailable. Run with --device cpu.")

    device = torch.device(args.device)
    seed = 42
    batch_size = 128
    width = 128

    torch.manual_seed(seed)
    torch.set_num_threads(1)

    preparation_metadata = json.loads((DATA_DIR / "preparation_metadata.json").read_text())

    with np.load(DATA_DIR / "prepared_data.npz") as data:
        # Load training and validation data only.
        x_train = torch.from_numpy(data["x_train"].copy()).to(device)
        y_train = torch.from_numpy(data["y_train"].copy()).to(device)
        x_validation = torch.from_numpy(data["x_validation"].copy()).to(device)
        y_validation = torch.from_numpy(data["y_validation"].copy()).to(device)

    cells = y_train.shape[1]
    model = Surrogate(cells=cells, width=width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=0.0001,
    )

    # Separate folders prevent CPU and MPS runs
    # from overwriting each other's results.
    output_dir = (PROJECT_ROOT / "artifacts" / args.device)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_validation_error = float("inf")
    best_epoch = 0
    best_state = None

    history = []

    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    print(f"Device: {device}", flush=True)
    print(f"Training examples: {len(x_train)}", flush=True)
    print(f"Model parameters: {parameter_count}", flush=True)

    synchronize(device)
    start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()

        # Shuffle the training examples every epoch.
        order = torch.randperm(len(x_train), device="cpu").to(device)
        total_loss = torch.zeros((), device=device)

        for indices in order.split(batch_size):
            batch_inputs = x_train[indices]
            batch_targets = y_train[indices]

            optimizer.zero_grad(set_to_none=True)
            log_probabilities = model(batch_inputs)

            # Cross-entropy against the solver's distribution.
            loss = -(batch_targets * log_probabilities).sum(dim=1).mean()
            loss.backward()
            optimizer.step()
            total_loss += (loss.detach() * len(indices))

        scheduler.step()
        training_loss = (total_loss / len(x_train)).item()

        model.eval()

        with torch.inference_mode():
            validation_probabilities = model(x_validation).exp()
            validation_error = (validation_probabilities - y_validation).abs().sum(dim=1).mean().item()

        if not (np.isfinite(training_loss) and np.isfinite(validation_error)):
            raise RuntimeError("Training produced a nonfinite value.")

        if validation_error < best_validation_error:
            best_validation_error = validation_error
            best_epoch = epoch

            # Clone the weights so later training updates
            # cannot change our saved best model.
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

        history.append({
            "epoch": epoch,
            "training_cross_entropy": training_loss,
            "validation_l1": validation_error,
        })

        if epoch == 1 or epoch % 25 == 0:
            print(
                f"Epoch {epoch:3d} | "
                f"Training loss: {training_loss:.5f} | "
                f"Validation L1: {validation_error:.5f}",
                flush=True,
            )

    synchronize(device)
    training_seconds = time.perf_counter() - start

    checkpoint = {
        "model_state_dict": best_state,
        "cells": cells,
        "width": width,
        "best_epoch": best_epoch,
        "best_validation_l1": best_validation_error,
        "preparation": preparation_metadata,
    }

    torch.save(checkpoint, output_dir / "model.pt",
    )

    report = {
        "device": str(device),
        "seed": seed,
        "epochs": args.epochs,
        "batch_size": batch_size,
        "cpu_threads": torch.get_num_threads(),
        "parameter_count": parameter_count,
        "torch_version": str(torch.__version__),
        "training_seconds": training_seconds,
        "best_epoch": best_epoch,
        "best_validation_l1": best_validation_error,
        "history": history,
    }

    (output_dir / "training.json").write_text(
        json.dumps(
            report,
            indent=2,
            allow_nan=False,
        ) + "\n"
    )

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best validation L1: {best_validation_error:.6f}")
    print(f"Training time: {training_seconds:.2f} seconds")
    print(f"Saved: {output_dir / 'model.pt'}")


if __name__ == "__main__":
    main()