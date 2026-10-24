import csv
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "optimization"


def load_timings(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing benchmark file: {path}")

    groups = {}

    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            key = (
                row["method"],
                int(row["batch_size"]),
            )

            groups.setdefault(key, []).append(
                float(row["milliseconds"])
            )

    return groups


def main():
    before = load_timings(RESULTS_DIR / "before.csv")
    after = load_timings(RESULTS_DIR / "after.csv")

    expected_keys = {
        (method, batch)
        for method in ["numerical", "analytical"]
        for batch in [1, 16, 64, 256]
    }

    if set(before) != expected_keys or set(after) != expected_keys:
        raise ValueError(
            "Files must contain both methods and all four batch sizes."
        )

    lines = [
        "| Method | Batch | Before median (ms) | "
        "After median (ms) | Speedup |",
        "|---|---:|---:|---:|---:|",
    ]

    for method in ["numerical", "analytical"]:
        for batch in [1, 16, 64, 256]:
            key = (method, batch)

            if len(before[key]) != len(after[key]):
                raise ValueError(
                    f"Measurement counts differ for {key}."
                )

            before_values = np.array(before[key])
            after_values = np.array(after[key])

            if not (
                np.isfinite(before_values).all()
                and np.isfinite(after_values).all()
                and (before_values > 0).all()
                and (after_values > 0).all()
            ):
                raise ValueError(f"Invalid timing values for {key}.")

            before_ms = float(np.median(before_values))
            after_ms = float(np.median(after_values))
            speedup = before_ms / after_ms

            lines.append(
                f"| {method} | {batch} | "
                f"{before_ms:.6f} | {after_ms:.6f} | "
                f"{speedup:.2f}x |"
            )

    table = "\n".join(lines)

    print(table)

    output_file = RESULTS_DIR / "comparison.md"
    output_file.write_text(table + "\n")

    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()