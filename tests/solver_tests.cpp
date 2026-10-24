#include "fluxfp/solver.hpp"

#include <cmath>
#include <iostream>
#include <numbers>
#include <stdexcept>

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_initial_density() {
    fluxfp::Config config;

    const double actual = fluxfp::exact_density(config.initial_mean, 0.0, config);
    const double expected = 1.0 / (config.initial_std * std::sqrt(2.0 * std::numbers::pi));
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

int main() {
    try {
        test_initial_density();
        test_long_term_density();
        test_total_probability();
        test_invalid_parameters();

        std::cout << "All analytical tests passed.\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Test failed: " << error.what() << "\n";

        return 1;
    }
}