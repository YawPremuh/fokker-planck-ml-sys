import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def main():
    metadata = json.loads((DATA_DIR / "dataset_metadata.json").read_text())

    data = np.loadtxt(DATA_DIR / "dataset.csv", delimiter=",", skiprows=1, ndmin=2)

    expected_shape = (metadata["samples"], 4 + metadata["cells"])

    if data.shape != expected_shape:
        raise ValueError(f"Expected shape {expected_shape}, got {data.shape}")

    if not np.isfinite(data).all():
        raise ValueError("Dataset contains nonfinite values.")

    inputs = data[:, :4]
    densities = data[:, 4:]

    dx = (metadata["right"] - metadata["left"]) / metadata["cells"]

    if (densities < 0.0).any():
        raise ValueError("Dataset contains negative densities.")

    masses = densities.sum(axis=1) * dx
    largest_mass_error = np.max(np.abs(masses - 1.0))

    if largest_mass_error > 1e-8:
        raise ValueError("Some distributions have invalid mass.")

    for column, name in enumerate(metadata["input_columns"]):
        lower, upper = metadata["parameter_ranges"][name]

        if ((inputs[:, column] < lower).any() or (inputs[:, column] > upper).any()):
            raise ValueError(f"{name} is outside its expected range.")

    unique_inputs = np.unique(inputs, axis=0)

    if len(unique_inputs) != len(inputs):
        raise ValueError("Duplicate input combinations found.")

    print(f"Examples: {len(data)}")
    print(f"Inputs per example: {inputs.shape[1]}")
    print(f"Density values per example: {densities.shape[1]}")
    print(f"Largest probability-mass error: {largest_mass_error:.3e}")
    print("Dataset checks passed.")


if __name__ == "__main__":
    main()