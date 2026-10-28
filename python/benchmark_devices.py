import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from model import Surrogate


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODEL_FILE = PROJECT_ROOT / "artifacts" / "cpu" / "model.pt"
OUTPUT_DIR = PROJECT_ROOT / "results" / "benchmark_devices"


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()


def main():
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable in this Python environment.")

    torch.set_num_threads(1)

    warmups = 10
    repeats = 100
    batch_sizes = [1, 16, 64, 256]
    checkpoint = torch.load(MODEL_FILE, map_location="cpu", weights_only=True)
    preparation = checkpoint["preparation"]
    dataset_hash = hashlib.sha256((DATA_DIR / "dataset.csv").read_bytes()).hexdigest()

    if dataset_hash != preparation["dataset_sha256"]:
        raise ValueError("Dataset differs from the model's training dataset.")

    with np.load(DATA_DIR / "prepared_data.npz") as data:
        test_indices = data["test_indices"].copy()
        raw_inputs = data["raw_inputs"].copy()
        test_targets = data["y_test"].astype(np.float64, copy=True)
        order = np.random.default_rng(preparation["seed"]).permutation(len(raw_inputs))
        expected_indices = order[int(0.85 * len(order)):]

        if not np.array_equal(test_indices, expected_indices):
            raise ValueError("Unexpected test split.")

        test_inputs = ((raw_inputs[test_indices] - np.array(preparation["input_mean"])) / np.array(preparation["input_std"])).astype(np.float32)

        if not np.allclose(test_inputs, data["x_test"], rtol=0.0, atol=1e-6):
            raise ValueError("Prepared inputs do not match.")

    raw_measurements = []
    summaries = []
    accuracy = {}
    cpu_predictions = None

    for device_name in ["cpu", "mps"]:
        device = torch.device(device_name)
        model = Surrogate(cells=checkpoint["cells"], width=checkpoint["width"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()

        with torch.inference_mode():
            predictions = (model(torch.from_numpy(test_inputs).to(device)).exp().cpu().numpy().astype(np.float64))

        if not np.isfinite(predictions).all():
            raise ValueError(f"{device_name} produced nonfinite predictions.")

        if (predictions < 0.0).any():
            raise ValueError(f"{device_name} produced negative probabilities.")

        mass_error = float(np.abs(predictions.sum(axis=1) - 1.0).max())

        if mass_error > 1e-5:
            raise ValueError(f"{device_name} probability normalization failed.")

        accuracy[device_name] = {
            "mean_l1_vs_solver": float(np.abs(predictions - test_targets).sum(axis=1).mean()),
            "maximum_mass_error": mass_error,
        }

        if device_name == "cpu":
            cpu_predictions = predictions
        else:
            accuracy["cpu_mps_agreement"] = {
                "maximum_cell_probability_difference": float(np.abs(predictions - cpu_predictions).max()),
                "mean_distribution_l1_difference": float(np.abs(predictions - cpu_predictions).sum(axis=1).mean()),
            }

        modes = (["host_to_host"] if device_name == "cpu" else ["resident", "host_to_host"])

        for mode in modes:
            for batch_size in batch_sizes:
                cpu_batches = []

                for iteration in range(warmups + repeats):
                    indices = (np.arange(batch_size) + iteration * batch_size) % len(test_inputs)
                    cpu_batches.append(torch.from_numpy(test_inputs[indices].copy()))

                if mode == "resident":
                    ready_batches = [batch.to(device) for batch in cpu_batches]
                else:
                    ready_batches = cpu_batches

                times_ms = []

                with torch.inference_mode():
                    for iteration, batch in enumerate(ready_batches):
                        synchronize(device)

                        start = time.perf_counter_ns()

                        if mode == "host_to_host":
                            inputs = batch.to(device)
                        else:
                            inputs = batch

                        probabilities = model(inputs).exp()

                        if mode == "host_to_host":
                            probabilities = probabilities.cpu()
                        synchronize(device)

                        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0

                        if iteration >= warmups:
                            times_ms.append(elapsed_ms)
                            raw_measurements.append({
                                "device": device_name,
                                "mode": mode,
                                "batch_size": batch_size,
                                "repeat": iteration - warmups,
                                "milliseconds": elapsed_ms,
                            })

                median_ms = float(np.median(times_ms))
                p95_ms = float(np.quantile(times_ms, 0.95))

                summaries.append({
                    "device": device_name,
                    "mode": mode,
                    "batch_size": batch_size,
                    "median_batch_ms": median_ms,
                    "p95_batch_ms": p95_ms,
                    "microseconds_per_forecast": (median_ms * 1000.0 / batch_size),
                    "forecasts_per_second": (batch_size * 1000.0 / median_ms),
                })

                print(
                    f"{device_name:3s} | "
                    f"{mode:12s} | "
                    f"batch {batch_size:3d} | "
                    f"median {median_ms:.4f} ms | "
                    f"p95 {p95_ms:.4f} ms",
                    flush=True,
                )

    cpu_medians = {row["batch_size"]: row["median_batch_ms"] for row in summaries if row["device"] == "cpu"}

    for row in summaries:
        row["speedup_vs_cpu"] = (cpu_medians[row["batch_size"]] / row["median_batch_ms"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with (OUTPUT_DIR / "raw_timings.csv").open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "device",
                "mode",
                "batch_size",
                "repeat",
                "milliseconds",
            ],
        )

        writer.writeheader()
        writer.writerows(raw_measurements)

    report = {
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "precision": "float32",
        "cpu_threads": torch.get_num_threads(),
        "warmups": warmups,
        "repeats": repeats,
        "checkpoint_epoch": checkpoint["best_epoch"],
        "checkpoint_sha256": hashlib.sha256(MODEL_FILE.read_bytes()).hexdigest(),
        "dataset_sha256": dataset_hash,
        "accuracy": accuracy,
        "summary": summaries,
        "timing_notes": {
            "included": ("Model forward pass and exponentiation. MPS timings include completion synchronization."),
            "resident": ("Inputs are already on the device. Outputs remain on the device."),
            "host_to_host": ("Inputs start on CPU and outputs return to CPU."),
            "excluded": ("File loading, model loading, input standardization, batch construction, and service queueing."),
        },
    }

    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(
            report,
            indent=2,
            allow_nan=False,
        ) + "\n"
    )

    table = [
        "| Device | Mode | Batch | Median (ms) | "
        "P95 (ms) | Speedup vs CPU |",
        "|---|---|---:|---:|---:|---:|",
    ]

    for row in summaries:
        table.append(
            f"| {row['device']} | {row['mode']} | "
            f"{row['batch_size']} | "
            f"{row['median_batch_ms']:.4f} | "
            f"{row['p95_batch_ms']:.4f} | "
            f"{row['speedup_vs_cpu']:.2f}x |"
        )

    table_text = "\n".join(table)

    (OUTPUT_DIR / "comparison.md").write_text(table_text + "\n")

    print("\n" + table_text)
    print("\nAccuracy checks:")
    print(json.dumps(accuracy, indent=2))
    print(f"\nSaved results to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()