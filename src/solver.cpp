#include "fluxfp/solver.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace fluxfp {

double exact_density(double x, double time, const Config& config) {
    if (
        !std::isfinite(x) ||
        !std::isfinite(time) ||
        !std::isfinite(config.theta) ||
        !std::isfinite(config.mean) ||
        !std::isfinite(config.sigma) ||
        !std::isfinite(config.initial_mean) ||
        !std::isfinite(config.initial_std)
    ) {
        throw std::invalid_argument("All inputs must be finite numbers.");
    }

    if (
        time < 0.0 ||
        config.theta <= 0.0 ||
        config.sigma <= 0.0 ||
        config.initial_std <= 0.0
    ) {
        throw std::invalid_argument("Time must be nonnegative. Theta, sigma, and initial_std must be positive.");
    }

    const double decay =std::exp(-config.theta * time);
    const double future_mean = config.mean + (config.initial_mean - config.mean) * decay;
    const double initial_variance = config.initial_std * config.initial_std;
    const double future_variance = initial_variance * decay * decay + (config.sigma * config.sigma / (2.0 * config.theta)) * (-std::expm1(-2.0 * config.theta * time));
    const double difference = x - future_mean;
    const double exponent = -(difference * difference) / (2.0 * future_variance);
    const double denominator = std::sqrt(2.0 * std::numbers::pi * future_variance);

    return std::exp(exponent) / denominator;
}

Result solve(double time, const Config& config, int cells) {
    (void)exact_density(config.initial_mean, time, config);

    if (cells < 4 || cells > 100000) {
        throw std::invalid_argument("Cells must be between 4 and 100000.");
    }

    const double left = -6.0;
    const double right = 6.0;
    const double dx = (right - left) / cells;
    const double diffusion = 0.5 * config.sigma * config.sigma;

    Result result;
    result.dx = dx;
    result.x.resize(cells);
    result.density.resize(cells);
    double initial_mass = 0.0;

    for (int i = 0; i < cells; ++i) {
        result.x[i] = left + (i + 0.5) * dx;
        result.density[i] =exact_density(result.x[i], 0.0, config);
        initial_mass += result.density[i] * dx;
    }

    if (!std::isfinite(initial_mass) || initial_mass <= 0.0) {
        throw std::runtime_error("The grid cannot represent the initial distribution.");
    }

    // Normalize only the initial distribution.
    for (double& value : result.density) {
        value /= initial_mass;
    }

    if (time == 0.0) {
        return result;
    }

    const double max_drift = config.theta * std::max(std::abs(config.mean - left), std::abs(config.mean - right));
    const double dt_limit = 0.9 / (2.0 * max_drift / dx + 2.0 * diffusion / (dx * dx));
    const double required_steps = std::ceil(time / dt_limit);

    if (
        !std::isfinite(required_steps) ||
        required_steps < 1.0 ||
        required_steps > 10000000.0
    ) {
        throw std::runtime_error("This configuration requires too many time steps.");
    }

    result.steps = static_cast<int>(required_steps);

    const double dt = time / result.steps;
    const double diffusion_over_dx = diffusion / dx;
    const double dt_over_dx = dt / dx;

    std::vector<double> left_coefficient(cells + 1, 0.0);
    std::vector<double> right_coefficient(cells + 1, 0.0);

    for (int face = 1; face < cells; ++face) {
        const double face_x = left + face * dx;
        const double velocity = config.theta * (config.mean - face_x);

        left_coefficient[face] = std::max(velocity, 0.0) + diffusion_over_dx;

        right_coefficient[face] = std::min(velocity, 0.0) - diffusion_over_dx;
    }

    std::vector<double> flux(cells + 1, 0.0);
    std::vector<double> next_density(cells);

    for (int step = 0; step < result.steps; ++step) {
        for (int face = 1; face < cells; ++face) {
            flux[face] = left_coefficient[face] * result.density[face - 1] + right_coefficient[face] * result.density[face];
        }

        for (int i = 0; i < cells; ++i) {
            next_density[i] = result.density[i] - dt_over_dx * (flux[i + 1] - flux[i]);
        }

        result.density.swap(next_density);
    }

    return result;
}

}  