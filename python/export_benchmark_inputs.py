import hashlib
import json
from pathlib import Path
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "benchmark_cpu"


def main():
    benchmark = json.loads((RESULTS_DIR / "summary.json").read_text())
    dataset_hash = hashlib.sha256((DATA_DIR / "dataset.csv").read_bytes()).hexdigest()
    model_hash = hashlib.sha256((
            PROJECT_ROOT
            / "artifacts"
            / "cpu"
            / "model.pt"
        ).read_bytes()
    ).hexdigest()

    if dataset_hash != benchmark["dataset_sha256"]:
        raise ValueError("Dataset differs from the neural benchmark.")

    if model_hash != benchmark["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from the neural benchmark.")

    metadata = json.loads((DATA_DIR / "dataset_metadata.json").read_text())
    preparation = json.loads((DATA_DIR / "preparation_metadata.json").read_text())

    if metadata["cells"] != 256:
        raise ValueError("This benchmark expects a 256-cell dataset.")

    if (
        metadata["left"] != -6.0
        or metadata["right"] != 6.0
        or metadata["initial_mean"] != 0.0
        or metadata["initial_std"] != 0.3
    ):
        raise ValueError("Dataset settings differ from the C++ benchmark.")

    with np.load(DATA_DIR / "prepared_data.npz") as data:
        test_indices = data["test_indices"]
        order = np.random.default_rng(preparation["seed"]).permutation(len(data["raw_inputs"]))
        expected_indices = order[int(0.85 * len(order)):]

        if not np.array_equal(test_indices, expected_indices):
            raise ValueError("Unexpected test split.")

        raw_test_inputs = data["raw_inputs"][test_indices]
        reconstructed = ((raw_test_inputs - data["input_mean"]) / data["input_std"]).astype(np.float32)

        if not np.allclose(reconstructed, data["x_test"], rtol=0.0, atol=1e-6):
            raise ValueError("Prepared test inputs do not match.")

    output_file = RESULTS_DIR / "test_parameters.csv"

    np.savetxt(
        output_file,
        raw_test_inputs,
        delimiter=",",
        header="theta,mean,sigma,time",
        comments="",
        fmt="%.17g",
    )

    print(f"Exported {len(raw_test_inputs)} test inputs.")
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()