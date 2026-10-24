import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from model import Surrogate


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_FILE = PROJECT_ROOT / "artifacts" / "cpu" / "model.pt"
RESULTS_DIR = PROJECT_ROOT / "results" / "evaluation_cpu"


def analytical_probabilities(parameters, grid, dx, metadata):
    """Evaluate the analytical density and convert to cell probabilities."""

    theta = parameters[:, 0]
    mean = parameters[:, 1]
    sigma = parameters[:, 2]
    time = parameters[:, 3]
    initial_mean = metadata["initial_mean"]
    initial_std = metadata["initial_std"]
    decay = np.exp(-theta * time)
    future_mean = (mean + (initial_mean - mean) * decay)
    future_variance = (initial_std**2 * decay**2 + (sigma**2 / (2.0 * theta)) * (-np.expm1(-2.0 * theta * time)))

    difference = grid[None, :] - future_mean[:, None]
    density = (np.exp(-(difference**2) / (2.0 * future_variance[:, None])) / np.sqrt(2.0 * np.pi * future_variance[:, None]))

    return density * dx


def calculate_metrics(predicted, reference, grid):
    """Compare distributions represented as cell probabilities."""

    difference = predicted - reference
    l1_per_example = np.abs(difference).sum(axis=1)
    predicted_mean = (predicted * grid).sum(axis=1)
    reference_mean = (reference * grid).sum(axis=1)
    tail_cells = grid > 1.0
    predicted_tail = predicted[:, tail_cells].sum(axis=1)
    reference_tail = reference[:, tail_cells].sum(axis=1)

    return {
        "mean_l1": float(l1_per_example.mean()),
        "median_l1": float(np.median(l1_per_example)),
        "p95_l1": float(np.quantile(l1_per_example, 0.95)),
        "max_l1": float(l1_per_example.max()),
        "mean_absolute_mean_error": float(np.abs(predicted_mean - reference_mean).mean()),
        "mean_absolute_tail_probability_error_x_gt_1": float(np.abs(predicted_tail - reference_tail).mean()),
    }


def main():
    torch.set_num_threads(1)

    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Cannot find {MODEL_FILE}. Train the CPU model first.")

    checkpoint = torch.load(MODEL_FILE, map_location="cpu", weights_only=True)
    preparation = checkpoint["preparation"]

    # Dataset verfication
    dataset_hash = hashlib.sha256((DATA_DIR / "dataset.csv").read_bytes()).hexdigest()

    if dataset_hash != preparation["dataset_sha256"]:
        raise ValueError("The dataset has changed since this model was trained.")

    metadata = json.loads((DATA_DIR / "dataset_metadata.json").read_text())

    with np.load(DATA_DIR / "prepared_data.npz") as data:
        test_indices = data["test_indices"].copy()
        raw_test_inputs = data["raw_inputs"][test_indices].copy()
        solver_probabilities = data["y_test"].astype(np.float64)

        # Check that the saved test split matches the training setup.
        generator = np.random.default_rng(preparation["seed"])
        order = generator.permutation(len(data["raw_inputs"]))
        expected_test_indices = order[int(0.85 * len(order)):]

        if not np.array_equal(test_indices, expected_test_indices):
            raise ValueError("The test split differs from the training setup.")

    cells = checkpoint["cells"]

    if (cells != metadata["cells"] or solver_probabilities.shape != (len(raw_test_inputs), cells)):
        raise ValueError("Model and dataset dimensions disagree.")

    input_mean = np.array(preparation["input_mean"])
    input_std = np.array(preparation["input_std"])
    scaled_test_inputs = ((raw_test_inputs - input_mean) / input_std).astype(np.float32)
    model = Surrogate(cells=cells, width=checkpoint["width"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.inference_mode():
        test_tensor = torch.from_numpy(scaled_test_inputs)
        model_probabilities = (model(test_tensor).exp().numpy().astype(np.float64))

    if not np.isfinite(model_probabilities).all():
        raise ValueError("Model produced invalid predictions.")

    dx = (metadata["right"] - metadata["left"]) / cells
    grid = (metadata["left"] + (np.arange(cells) + 0.5) * dx)
    exact_probabilities = analytical_probabilities(raw_test_inputs, grid, dx,metadata)

    report = {
        "test_examples": len(raw_test_inputs),
        "device": "cpu",
        "checkpoint_epoch": checkpoint["best_epoch"],
        "dataset_sha256": dataset_hash,
        "checkpoint_sha256": hashlib.sha256(MODEL_FILE.read_bytes()).hexdigest(),
        "model_vs_solver": calculate_metrics(model_probabilities, solver_probabilities, grid),
        "solver_vs_analytical": calculate_metrics(solver_probabilities, exact_probabilities, grid),
        "model_vs_analytical": calculate_metrics(model_probabilities, exact_probabilities, grid),
        "model_probability_checks": {
            "minimum_probability": float(model_probabilities.min()),
            "maximum_mass_error": float(np.abs(model_probabilities.sum(axis=1) - 1.0).max()),
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    # Save predictions 
    np.savez_compressed(
        RESULTS_DIR / "predictions.npz",
        grid=grid,
        dx=np.array(dx),
        parameters=raw_test_inputs,
        test_indices=test_indices,
        model_probabilities=model_probabilities,
        solver_probabilities=solver_probabilities,
        analytical_probabilities=exact_probabilities,
    )

    print(f"Test examples: {len(raw_test_inputs)}")
    print(f"Loaded checkpoint: epoch {checkpoint['best_epoch']}")

    for comparison in ["model_vs_solver", "solver_vs_analytical", "model_vs_analytical"]:
        values = report[comparison]

        print(f"\n{comparison}")
        print(f"  Mean L1: {values['mean_l1']:.6f}")
        print(f"  Median L1: {values['median_l1']:.6f}")
        print(f"  95th-percentile L1: {values['p95_l1']:.6f}")
        print(f"  Maximum L1: {values['max_l1']:.6f}")
        print(f"  Mean absolute mean error: {values['mean_absolute_mean_error']:.6f}")
        print(f"  Mean absolute tail probability error: {values['mean_absolute_tail_probability_error_x_gt_1']:.6f}")

    checks = report["model_probability_checks"]

    print(f"\nMaximum model probability-mass error: {checks['maximum_mass_error']:.3e}")
    print(f"\nSaved evaluation to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()