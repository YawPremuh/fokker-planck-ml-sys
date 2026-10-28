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
OUTPUT_DIR = PROJECT_ROOT / "results" / "benchmark_cpu"


def main():
    torch.set_num_threads(1)

    batch_sizes = [1, 16, 64, 256]
    warmups = 10
    repeats = 100

    checkpoint = torch.load(MODEL_FILE, map_location="cpu", weights_only=True)
    dataset_hash = hashlib.sha256((DATA_DIR / "dataset.csv").read_bytes()).hexdigest()

    if dataset_hash != checkpoint["preparation"]["dataset_sha256"]:
        raise ValueError("The dataset differs from the model's training dataset.")

    with np.load(DATA_DIR / "prepared_data.npz") as data:
        test_inputs = data["x_test"].astype(np.float32, copy=True)
        input_mean = data["input_mean"].copy()
        input_std = data["input_std"].copy()
        test_indices = data["test_indices"].copy()
        sample_count = len(data["raw_inputs"])

    preparation = checkpoint["preparation"]

    if not (
        np.allclose(input_mean, preparation["input_mean"], rtol=0.0, atol=1e-12)
        and np.allclose(input_std, preparation["input_std"], rtol=0.0, atol=1e-12,)
    ):
        raise ValueError("Input scaling differs from the model's training setup.")

    order = np.random.default_rng(preparation["seed"]).permutation(sample_count)
    expected_test_indices = order[int(0.85 * sample_count):]

    if not np.array_equal(test_indices, expected_test_indices):
        raise ValueError("The test split differs from the model's training setup.")

    model = Surrogate(cells=checkpoint["cells"], width=checkpoint["width"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    raw_measurements = []
    summaries = []

    print("Device: CPU")
    print(f"CPU threads: {torch.get_num_threads()}")
    print(f"Warmup batches per size: {warmups}")
    print(f"Measured batches per size: {repeats}")

    with torch.inference_mode():
        example_probabilities = model(torch.from_numpy(test_inputs)).exp()

        if not torch.isfinite(example_probabilities).all().item():
            raise ValueError("Model produced nonfinite probabilities.")

        if not torch.allclose(
            example_probabilities.sum(dim=1),
            torch.ones(len(test_inputs)),
            atol=1e-6,
            rtol=0.0,
        ):
            raise ValueError("Model probabilities do not sum to one.")

        for batch_size in batch_sizes:
            batches = []

            for repeat in range(warmups + repeats):
                indices = (np.arange(batch_size) + repeat * batch_size) % len(test_inputs)
                batch = torch.from_numpy(test_inputs[indices].copy())
                batches.append(batch)

            for batch in batches[:warmups]:
                model(batch).exp()

            elapsed_times_ms = []

            for repeat, batch in enumerate(batches[warmups:]):
                start = time.perf_counter_ns()
                probabilities = model(batch).exp()
                elapsed_ns = time.perf_counter_ns() - start
                elapsed_ms = elapsed_ns / 1_000_000.0
                elapsed_times_ms.append(elapsed_ms)
                raw_measurements.append({
                    "batch_size": batch_size,
                    "repeat": repeat,
                    "milliseconds": elapsed_ms,
                })

            median_ms = float(np.median(elapsed_times_ms))
            p95_ms = float(np.quantile(elapsed_times_ms, 0.95))
            forecasts_per_second = (batch_size * 1000.0 / median_ms)
            microseconds_per_forecast = (median_ms * 1000.0 / batch_size)
            
            summary = {
                "batch_size": batch_size,
                "median_batch_ms": median_ms,
                "p95_batch_ms": p95_ms,
                "microseconds_per_forecast": microseconds_per_forecast,
                "forecasts_per_second": forecasts_per_second,
            }

            summaries.append(summary)

            print(
                f"\nBatch size: {batch_size}\n"
                f"  Median batch latency: {median_ms:.4f} ms\n"
                f"  P95 batch latency: {p95_ms:.4f} ms\n"
                f"  Time per forecast: "
                f"{microseconds_per_forecast:.2f} microseconds\n"
                f"  Throughput: "
                f"{forecasts_per_second:,.0f} forecasts/second"
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with (OUTPUT_DIR / "raw_timings.csv").open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "batch_size",
                "repeat",
                "milliseconds",
            ],
        )

        writer.writeheader()
        writer.writerows(raw_measurements)

    report = {
        "device": "cpu",
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "precision": "float32",
        "cpu_threads": torch.get_num_threads(),
        "warmups_per_batch_size": warmups,
        "repeats_per_batch_size": repeats,
        "checkpoint_epoch": checkpoint["best_epoch"],
        "checkpoint_sha256": hashlib.sha256(
            MODEL_FILE.read_bytes()
        ).hexdigest(),
        "dataset_sha256": dataset_hash,
        "timed_operation": (
            "CPU model forward pass and exponentiation "
            "from standardized inputs to cell probabilities"
        ),
        "excluded": [
            "File loading",
            "Model construction and checkpoint loading",
            "Input standardization",
            "Input batch creation",
            "Service queueing",
        ],
        "summary": summaries,
    }

    (OUTPUT_DIR / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    print(f"\nSaved benchmark results to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()