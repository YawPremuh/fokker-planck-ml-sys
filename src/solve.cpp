#include "fluxfp/solver.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>

namespace fluxfp {

double exact_density(
    double x,
    double time,
    const Config& config
) {
    //reject invalid inputs that would make calc invalid
    if (
        !std::isfinite(x) ||
        !std::isfinite(time) ||
        !std::isfinite(config.theta) ||
        !std::isfinite(config.mean) ||
        !std::isfinite(config.sigma) ||
        !std::isfinite(config.initial_mean) ||
        !std::isfinite(config.initial_std)
    ) {
        throw std::invalid_argument(
            "All your inputs must be finite numbers."
        );
    }

    if (
        time < 0.0 ||
        config.theta <= 0.0 ||
        config.sigma <= 0.0 ||
        config.initial_std <= 0.0
    ) {
        throw std::invalid_argument(
            "Time must be not be negative. "
            "Theta, sigma, and initial_std must be positive."
        );
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

}