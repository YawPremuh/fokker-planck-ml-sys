#include "fluxfp/solver.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <numbers>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

constexpr int cells = 256;
constexpr double left = -6.0;
constexpr double right = 6.0;
constexpr double dx = (right - left) / cells;

struct Example {
    fluxfp::Config config;
    double time;
};

std::vector<Example> load_examples(
    const std::string& filename
) {
    std::ifstream input(filename);

    if (!input) {
        throw std::runtime_error(
            "Cannot open benchmark inputs. "
            "Run python/export_benchmark_inputs.py first."
        );
    }

    std::string line;

    // Skip the CSV header.
    std::getline(input, line);

    std::vector<Example> examples;

    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }

        std::replace(
            line.begin(),
            line.end(),
            ',',
            ' '
        );

        std::istringstream row(line);

        Example example;

        if (!(
            row
            >> example.config.theta
            >> example.config.mean
            >> example.config.sigma
            >> example.time
        )) {
            throw std::runtime_error(
                "Invalid benchmark input row."
            );
        }

        // Validate the parameters before any timing.
        (void)fluxfp::exact_density(
            0.0,
            example.time,
            example.config
        );

        examples.push_back(example);
    }

    if (examples.empty()) {
        throw std::runtime_error(
            "No benchmark inputs found."
        );
    }

    return examples;
}

std::vector<double> analytical_probabilities(
    const Example& example
) {
    const auto& config = example.config;

    // Calculate distribution parameters once per forecast,
    // rather than repeating them for every grid position.
    const double decay =
        std::exp(-config.theta * example.time);

    const double mean =
        config.mean
        + (config.initial_mean - config.mean) * decay;

    const double variance =
        config.initial_std * config.initial_std
        * decay * decay
        + (
            config.sigma * config.sigma
            / (2.0 * config.theta)
        ) * (
            -std::expm1(
                -2.0 * config.theta * example.time
            )
        );

    const double normalization =
        1.0 / std::sqrt(2.0 * std::numbers::pi * variance);

    std::vector<double> probabilities(cells);

    for (int i = 0; i < cells; ++i) {
        const double x =
            left + (i + 0.5) * dx;

        const double difference = x - mean;

        probabilities[i] =
            normalization
            * std::exp(
                -(difference * difference)
                / (2.0 * variance)
            )
            * dx;
    }

    return probabilities;
}

std::vector<double> numerical_probabilities(
    const Example& example
) {
    auto result = fluxfp::solve(
        example.time,
        example.config,
        cells
    );

    // Match the neural model's output: cell probabilities.
    for (double& value : result.density) {
        value *= result.dx;
    }

    return result.density;
}

double quantile(
    std::vector<double> values,
    double fraction
) {
    std::sort(values.begin(), values.end());

    const double position =
        fraction * (values.size() - 1);

    const auto lower =
        static_cast<std::size_t>(std::floor(position));

    const auto upper =
        static_cast<std::size_t>(std::ceil(position));

    const double weight = position - lower;

    return (
        values[lower] * (1.0 - weight)
        + values[upper] * weight
    );
}

int main() {
    try {
        const auto examples = load_examples(
            "results/benchmark_cpu/test_parameters.csv"
        );

        const int warmups = 10;
        const int repeats = 100;

        // Verify the optimized analytical implementation
        // against our existing reference before timing it.
        for (const auto& example : examples) {
            const auto probabilities =
                analytical_probabilities(example);

            for (int i = 0; i < cells; ++i) {
                const double x =
                    left + (i + 0.5) * dx;

                const double reference =
                    fluxfp::exact_density(
                        x,
                        example.time,
                        example.config
                    ) * dx;

                if (
                    !std::isfinite(probabilities[i])
                    || std::abs(
                        probabilities[i] - reference
                    ) > 1e-12
                ) {
                    throw std::runtime_error(
                        "Analytical benchmark verification failed."
                    );
                }
            }
        }

        std::filesystem::create_directories(
            "results/benchmark_cpp"
        );

        std::ofstream output(
            "results/benchmark_cpp/raw_timings.csv"
        );

        if (!output) {
            throw std::runtime_error(
                "Cannot create benchmark output."
            );
        }

        output << std::setprecision(17);

        output
            << "method,batch_size,repeat,milliseconds\n";

        // Using the predictions after timing ensures their
        // computed values are observable.
        double checksum = 0.0;

        for (const std::string method :
             {"numerical", "analytical"}) {
            for (int batch_size : {1, 16, 64, 256}) {
                std::vector<double> measured_times;

                for (
                    int iteration = 0;
                    iteration < warmups + repeats;
                    ++iteration
                ) {
                    // Prepare the input selection outside the timer.
                    std::vector<const Example*> batch;

                    batch.reserve(batch_size);

                    for (int i = 0; i < batch_size; ++i) {
                        const std::size_t index =
                            (
                                iteration * batch_size + i
                            ) % examples.size();

                        batch.push_back(&examples[index]);
                    }

                    std::vector<std::vector<double>> predictions;
                    predictions.reserve(batch_size);

                    const auto start =
                        std::chrono::steady_clock::now();

                    if (method == "numerical") {
                        for (const Example* example : batch) {
                            predictions.push_back(
                                numerical_probabilities(*example)
                            );
                        }
                    } else {
                        for (const Example* example : batch) {
                            predictions.push_back(
                                analytical_probabilities(*example)
                            );
                        }
                    }

                    const auto finish =
                        std::chrono::steady_clock::now();

                    const double elapsed_ms =
                        std::chrono::duration<double, std::milli>(
                            finish - start
                        ).count();

                    // Validate and consume outputs outside the timer.
                    for (const auto& probabilities : predictions) {
                        for (
                            std::size_t i = 0;
                            i < probabilities.size();
                            ++i
                        ) {
                            if (
                                !std::isfinite(probabilities[i])
                                || probabilities[i] < 0.0
                            ) {
                                throw std::runtime_error(
                                    "Invalid benchmark prediction."
                                );
                            }

                            checksum +=
                                (i + 1) * probabilities[i];
                        }
                    }

                    if (iteration >= warmups) {
                        const int repeat = iteration - warmups;

                        measured_times.push_back(elapsed_ms);

                        output
                            << method << ","
                            << batch_size << ","
                            << repeat << ","
                            << elapsed_ms << "\n";
                    }
                }

                const double median_ms =
                    quantile(measured_times, 0.5);

                const double p95_ms =
                    quantile(measured_times, 0.95);

                const double throughput =
                    batch_size * 1000.0 / median_ms;

                const double microseconds_per_forecast =
                    median_ms * 1000.0 / batch_size;

                std::cout
                    << std::fixed
                    << std::setprecision(4)
                    << "\nMethod: " << method
                    << "\nBatch size: " << batch_size
                    << "\n  Median batch latency: "
                    << median_ms << " ms"
                    << "\n  P95 batch latency: "
                    << p95_ms << " ms"
                    << "\n  Time per forecast: "
                    << microseconds_per_forecast
                    << " microseconds"
                    << "\n  Throughput: "
                    << throughput
                    << " forecasts/second\n";
            }
        }

        output.close();

        if (!output) {
            throw std::runtime_error(
                "Failed to write benchmark results."
            );
        }

        std::cout
            << "\nOutput checksum: " << checksum
            << "\nSaved results/benchmark_cpp/raw_timings.csv\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "Error: "
            << error.what()
            << "\n";

        return 1;
    }
}