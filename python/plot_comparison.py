from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "results" / "comparison.csv"
OUTPUT_FILE = PROJECT_ROOT / "results" / "comparison.png"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Cannot find {INPUT_FILE}. Run ./build/fluxfp first.")

    # Read the CSV. Its first row provides the column names.
    data = np.genfromtxt(INPUT_FILE, delimiter=",", names=True)
    x = data["x"]
    numerical = data["numerical_density"]
    analytical = data["analytical_density"]

    if not (
        np.isfinite(x).all()
        and np.isfinite(numerical).all()
        and np.isfinite(analytical).all()
    ):
        raise ValueError("The CSV contains invalid numbers.")

    dx = x[1] - x[0]
    error = numerical - analytical
    l1_error = np.sum(np.abs(error)) * dx
    numerical_mass = np.sum(numerical) * dx

    # Two vertically stacked plots.
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(9, 7),
        sharex=True,
        constrained_layout=True,
    )

    axes[0].plot(
        x,
        analytical,
        label="Analytical reference",
        color="black",
        linewidth=2,
    )

    axes[0].plot(
        x,
        numerical,
        label="Numerical solver",
        color="tab:blue",
        linestyle="--",
        linewidth=2,
    )

    axes[0].set_title("Fokker–Planck numerical solver vs analytical reference")
    axes[0].set_ylabel("Probability density")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    # Bottom plot: show the difference directly.
    axes[1].plot(
        x,
        error,
        color="tab:red",
        label="Numerical − analytical",
    )

    axes[1].axhline(
        0.0,
        color="black",
        linewidth=1,
    )

    axes[1].set_xlabel("State x")
    axes[1].set_ylabel("Density difference")
    axes[1].set_title(f"L1 error = {l1_error:.6f}")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    figure.savefig(OUTPUT_FILE, dpi=160)
    plt.close(figure)

    print(f"Numerical total probability: {numerical_mass:.10f}")
    print(f"L1 error: {l1_error:.10f}")
    print(f"Saved plot: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()