from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = (PROJECT_ROOT / "results" / "evaluation_cpu")


def main():
    input_file = RESULTS_DIR / "predictions.npz"

    if not input_file.exists():
        raise FileNotFoundError("Run python python/evaluate.py first.")

    with np.load(input_file) as data:
        grid = data["grid"].copy()
        dx = float(data["dx"].item())
        parameters = data["parameters"].copy()
        test_indices = data["test_indices"].copy()
        model_probabilities = data["model_probabilities"].copy()
        solver_probabilities = data["solver_probabilities"].copy()
        analytical_probabilities = data["analytical_probabilities"].copy()

    model_l1 = np.abs(model_probabilities - analytical_probabilities).sum(axis=1)
    solver_l1 = np.abs(solver_probabilities - analytical_probabilities).sum(axis=1)
    order = np.argsort(model_l1)
    selected = [int(order[0]), int(order[len(order) // 2]), int(order[-1]),]
    labels = ["Lowest-error example", "Middle-ranked example", "Highest-error example"]
    model_density = model_probabilities / dx
    solver_density = solver_probabilities / dx
    analytical_density = analytical_probabilities / dx
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True,)

    for column, (index, label) in enumerate(zip(selected, labels)):
        theta, mean, sigma, time = parameters[index]
        top = axes[0, column]
        bottom = axes[1, column]
        top.plot(grid, analytical_density[index], color="black", linewidth=2, label="Analytical reference")
        top.plot(grid, solver_density[index], color="tab:blue", linestyle="--", linewidth=2, label="Numerical solver",)
        top.plot(grid, model_density[index], color="tab:orange", linestyle=":", linewidth=2, label="Neural surrogate",)
        top.set_title(
            f"{label}\n"
            f"theta={theta:.2f}, mean={mean:.2f}\n"
            f"sigma={sigma:.2f}, time={time:.2f}"
        )

        top.set_xlabel("State x")
        top.set_ylabel("Probability density")
        top.grid(alpha=0.25)
        top.legend(fontsize=8)

        bottom.plot(
            grid,
            model_density[index] - analytical_density[index],
            color="tab:orange",
            label="Model − analytical",
        )

        bottom.plot(
            grid,
            solver_density[index] - analytical_density[index],
            color="tab:blue",
            linestyle="--",
            label="Solver − analytical",
        )

        bottom.axhline(
            0.0,
            color="black",
            linewidth=1,
        )

        bottom.set_title(
            f"Model L1: {model_l1[index]:.5f}\n"
            f"Solver L1: {solver_l1[index]:.5f}"
        )

        bottom.set_xlabel("State x")
        bottom.set_ylabel("Density difference")
        bottom.grid(alpha=0.25)
        bottom.legend(fontsize=8)

        # Use the same horizontal range within each column.
        bottom.set_xlim(top.get_xlim())

        print(
            f"{label}: "
            f"dataset row {test_indices[index]}, "
            f"model L1={model_l1[index]:.6f}, "
            f"solver L1={solver_l1[index]:.6f}"
        )

    figure.suptitle(
        "Held-out predictions ranked by model L1 error "
        "against the analytical reference",
        fontsize=14,
    )

    comparison_file = RESULTS_DIR / "prediction_examples.png"

    figure.savefig(comparison_file, dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5),constrained_layout=True)
    bins = np.linspace(0.0,max(model_l1.max(), solver_l1.max()) * 1.05, 26)

    axis.hist(
        solver_l1,
        bins=bins,
        alpha=0.55,
        color="tab:blue",
        label="Solver vs analytical",
    )

    axis.hist(
        model_l1,
        bins=bins,
        alpha=0.55,
        color="tab:orange",
        label="Model vs analytical",
    )

    axis.set_title(f"Error distribution across {len(model_l1)} test examples")
    axis.set_xlabel("L1 distribution error")
    axis.set_ylabel("Number of examples")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)

    histogram_file = RESULTS_DIR / "test_error_distribution.png"

    figure.savefig(histogram_file, dpi=160)
    plt.close(figure)

    print(f"\nSaved: {comparison_file}")
    print(f"Saved: {histogram_file}")


if __name__ == "__main__":
    main()