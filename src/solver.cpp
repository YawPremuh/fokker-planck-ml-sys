#include "fluxfp/solver.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <algorithm>

namespace fluxfp {

double exact_density(double x, double time,const Config& config) {
    //reject invalid inputs that would make calc invalid
    if (!std::isfinite(x) ||
        !std::isfinite(time) ||
        !std::isfinite(config.theta) ||
        !std::isfinite(config.mean) ||
        !std::isfinite(config.sigma) ||
        !std::isfinite(config.initial_mean) ||
        !std::isfinite(config.initial_std)
    ) {
        throw std::invalid_argument("All your inputs must be finite numbers.");
    }

    if (time < 0.0 || config.theta <= 0.0 || config.sigma <= 0.0 || config.initial_std <= 0.0) {
        throw std::invalid_argument("Time must be not be negative. Theta, sigma, and initial_std must be positive.");
    }

    //how much influence the initial condition still has
    const double decay = std::exp(-config.theta * time);

    //mean of the distribution at the requested time
    const double future_mean = config.mean + (config.initial_mean - config.mean) * decay;

    //variance gotten from the initial distribution
    const double initial_variance = config.initial_std * config.initial_std;

    //variance at the requested time
    const double future_variance = initial_variance * decay * decay + (config.sigma * config.sigma / (2.0 * config.theta)) * (-std::expm1(-2.0 * config.theta * time));

    //evaluate the gaussian density at position x.
    const double difference = x - future_mean;
    const double exponent = -(difference * difference) / (2.0 * future_variance);
    const double denominator = std::sqrt(2.0 * std::numbers::pi * future_variance);

    return std::exp(exponent) / denominator;
}

Result solve(double time, const Config& config, int cells) {
    // Reuse the analytical function's parameter validation.
    // This checks time and the process parameters.
    (void)exact_density(config.initial_mean, time, config);

    if (cells < 4 || cells > 100000) {
        throw std::invalid_argument("Cells must be between 4 and 100000.");
    }

    const double left = -6.0;
    const double right = 6.0;
    const double dx = (right - left) / cells;

    // D is the diffusion coefficient in the PDE.
    const double diffusion = 0.5 * config.sigma * config.sigma;

    Result result;
    result.dx = dx;
    result.x.resize(cells);
    result.density.resize(cells);

    // Create the initial distribution.
    double initial_mass = 0.0;

    for (int i = 0; i < cells; ++i) {
        result.x[i] = left + (i + 0.5) * dx;
        result.density[i] = exact_density(result.x[i], 0.0, config);
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

    // Largest drift speed anywhere on this interval.
    const double max_drift = config.theta * std::max(std::abs(config.mean - left), std::abs(config.mean - right));

    // A conservative bound for stable explicit time stepping.
    const double dt_limit = 0.9 / (2.0 * max_drift / dx + 2.0 * diffusion / (dx * dx));
    const double required_steps = std::ceil(time / dt_limit);

    if (!std::isfinite(required_steps) || required_steps < 1.0 || required_steps > 10000000.0
    ) {
        throw std::runtime_error("This configuration requires too many time steps.");
    }

    result.steps = static_cast<int>(required_steps);

    // Adjust dt so we finish exactly at the requested time.
    const double dt = time / result.steps;

    // N cells have N+1 faces.
    std::vector<double> flux(cells + 1, 0.0);

    // Keep old and new densities separate.
    std::vector<double> next_density(cells);

    for (int step = 0; step < result.steps; ++step) {
        // No probability crosses the outer boundaries.
        flux[0] = 0.0;
        flux[cells] = 0.0;

        // Calculate flow across internal faces.
        for (int face = 1; face < cells; ++face) {
            const double face_x = left + face * dx;
            const double velocity = config.theta * (config.mean - face_x);

            // Use the density on the upstream side.
            const double upstream_density = velocity >= 0.0 ? result.density[face - 1] : result.density[face];
            const double drift_flux = velocity * upstream_density;
            const double density_gradient = (result.density[face] - result.density[face - 1]) / dx;
            const double diffusion_flux = -diffusion * density_gradient;

            flux[face] = drift_flux + diffusion_flux;
        }

        // Update every cell using the same old distribution.
        for (int i = 0; i < cells; ++i) {
            next_density[i] = result.density[i] - (dt / dx) * (flux[i + 1] - flux[i]);
        }

        result.density.swap(next_density);
    }

    return result;
}

}