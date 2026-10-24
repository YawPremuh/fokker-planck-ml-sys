#include "fluxfp/solver.hpp"

#include <cmath>
#include <numbers>
#include <iostream>
#include <stdexcept>

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_initial_density() {
    fluxfp::Config config;

    const double actual = fluxfp::exact_density(config.initial_mean, 0.0, config);
    const double expected = 1.0 / (config.initial_std * std::sqrt(2.0 *std::numbers::pi));
    require(std::abs(actual - expected) < 1e-12, "Initial density test failed.");
}

void test_long_term_density() {
    fluxfp::Config config;

    const double stationary_variance = config.sigma * config.sigma / (2.0 * config.theta);
    const double expected = 1.0 / std::sqrt(2.0 * std::numbers::pi * stationary_variance);
    const double actual = fluxfp::exact_density(config.mean, 30.0, config);
    require(std::abs(actual - expected) < 1e-10, "Long-term density test failed.");
}

void test_total_probability() {
    fluxfp::Config config;

    const int cells = 2000;
    const double left = -6.0;
    const double right = 6.0;
    const double dx = (right - left) / cells;

    double total = 0.0;

    for (int i = 0; i < cells; ++i) {
        const double x = left + (i + 0.5) * dx;
        const double density = fluxfp::exact_density(x, 1.0, config);
        require(density >= 0.0, "Density must not be negative.");
        total += density * dx;
    }

    require(std::abs(total - 1.0) < 1e-8, "Total probability is not approximately one.");
}

void test_invalid_parameters() {
    fluxfp::Config config;
    config.sigma = -0.8;
    bool rejected = false;

    try {
        fluxfp::exact_density(0.0, 1.0, config);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }

    require(rejected, "The negative noise strength should be rejected."
    );
}

void test_numerical_initial_condition() {
    fluxfp::Config config;
    const auto result = fluxfp::solve(0.0, config, 400);
    require(result.steps == 0, "Zero forecast time should require no time steps.");
    double error = 0.0;

    for (std::size_t i = 0; i < result.x.size(); ++i) {
        const double expected = fluxfp::exact_density(result.x[i], 0.0, config);
        error += std::abs(result.density[i] - expected) * result.dx;
    }

    require(error < 1e-10, "Numerical initial distribution is incorrect.");
}

void test_numerical_convergence() {
    fluxfp::Config config;
    const double time = 1.0;
    double previous_error = 10.0;

    for (int cells : {100, 200, 400, 800}) {
        const auto result = fluxfp::solve(time, config, cells);

        double total_probability = 0.0;
        double error = 0.0;

        for (std::size_t i = 0; i < result.x.size(); ++i) {
            require(std::isfinite(result.density[i]), "Numerical density must be finite.");
            require(result.density[i] >= 0.0, "Numerical density must not be negative.");

            total_probability += result.density[i] * result.dx;
            const double exact = fluxfp::exact_density(result.x[i], time, config);
            error += std::abs(result.density[i] - exact) * result.dx;
        }

        require(std::abs(total_probability - 1.0) < 1e-10, "Numerical solver did not conserve probability.");
        require(error < previous_error, "Refining the grid did not reduce error.");
        std::cout << cells << " cells: L1 error = " << error << "\n";
        previous_error = error;
    }

    require(previous_error < 0.015, "Fine-grid error is larger than expected.");
}

int main() {
    try {
        test_initial_density();
        test_long_term_density();
        test_total_probability();
        test_invalid_parameters();
        test_numerical_initial_condition();
        test_numerical_convergence();

        std::cout << "All analytical and numerical tests passed.\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Test failed: " << error.what() << "\n";

        return 1;
    }
}