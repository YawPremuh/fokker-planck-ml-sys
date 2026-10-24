import hashlib
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def main():
    csv_path = DATA_DIR / "dataset.csv"
    metadata = json.loads((DATA_DIR / "dataset_metadata.json").read_text())

    data = np.loadtxt(
        csv_path,
        delimiter=",",
        skiprows=1,
        ndmin=2,
    )

    inputs = data[:, :4]
    densities = data[:, 4:]

    if data.shape != (metadata["samples"], 4 + metadata["cells"]):
        raise ValueError("Dataset shape disagrees with metadata.")

    if not np.isfinite(data).all():
        raise ValueError("Dataset contains invalid numbers.")

    if len(np.unique(inputs, axis=0)) != len(inputs):
        raise ValueError("Duplicate input combinations found.")

    dx = (metadata["right"] - metadata["left"]) / metadata["cells"]

    # Convert density values into approximate cell probabilities.
    probabilities = densities * dx

    if (probabilities < 0.0).any():
        raise ValueError("Negative probabilities found.")

    if not np.allclose(
        probabilities.sum(axis=1),
        1.0,
        atol=1e-8,
        rtol=0.0,
    ):
        raise ValueError("Probabilities do not sum to one.")

    # Shuffle row indices reproducibly.
    seed = 42
    generator = np.random.default_rng(seed)
    indices = generator.permutation(len(inputs))

    train_end = int(0.70 * len(inputs))
    validation_end = int(0.85 * len(inputs))

    train_indices = indices[:train_end]
    validation_indices = indices[train_end:validation_end]
    test_indices = indices[validation_end:]

    # Calculate scaling statistics using training inputs ONLY.
    input_mean = inputs[train_indices].mean(axis=0)
    input_std = inputs[train_indices].std(axis=0)

    if (input_std <= 0.0).any():
        raise ValueError("A training input has zero variation.")

    scaled_inputs = (inputs - input_mean) / input_std
    output_path = DATA_DIR / "prepared_data.npz"

    np.savez_compressed(
        output_path,
        x_train=scaled_inputs[train_indices].astype(np.float32),
        y_train=probabilities[train_indices].astype(np.float32),

        x_validation=scaled_inputs[validation_indices].astype(
            np.float32
        ),
        y_validation=probabilities[validation_indices].astype(
            np.float32
        ),

        x_test=scaled_inputs[test_indices].astype(np.float32),
        y_test=probabilities[test_indices].astype(np.float32),

        raw_inputs=inputs,
        input_mean=input_mean,
        input_std=input_std,

        train_indices=train_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,

        dx=np.array(dx),
    )

    preparation_metadata = {
        "seed": seed,
        "training_examples": len(train_indices),
        "validation_examples": len(validation_indices),
        "test_examples": len(test_indices),
        "input_columns": metadata["input_columns"],
        "input_mean": input_mean.tolist(),
        "input_std": input_std.tolist(),
        "target_type": "cell_probability",
        "cells": metadata["cells"],
        "dx": dx,
        "dataset_sha256": hashlib.sha256(
            csv_path.read_bytes()
        ).hexdigest(),
    }

    (DATA_DIR / "preparation_metadata.json").write_text(
        json.dumps(preparation_metadata, indent=2) + "\n"
    )

    print(f"Training examples: {len(train_indices)}")
    print(f"Validation examples: {len(validation_indices)}")
    print(f"Test examples: {len(test_indices)}")
    print(f"Training input shape: {scaled_inputs[train_indices].shape}")
    print(f"Training target shape: {probabilities[train_indices].shape}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()