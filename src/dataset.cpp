#include "fluxfp/solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>

int main() {
    try {
        const int samples = 3000;
        const int cells = 256;
        const unsigned int seed = 42;

        std::filesystem::create_directories("data");
        std::ofstream output("data/dataset.csv");

        if (!output) {
            throw std::runtime_error("Could not open data/dataset.csv");
        }

        output << std::setprecision(17);

        // Write the column names.
        output << "theta,mean,sigma,time";

        for (int i = 0; i < cells; ++i) {
            output << ",p" << i;
        }

        output << "\n";

        // A fixed seed makes the sampling repeatable
        // with the same implementation.
        std::mt19937 generator(seed);
        std::uniform_real_distribution<double> theta_distribution(0.5, 2.0);
        std::uniform_real_distribution<double> mean_distribution(-1.0, 1.0);
        std::uniform_real_distribution<double> sigma_distribution(0.4, 1.0);
        std::uniform_real_distribution<double> time_distribution(0.1, 2.0);

        double largest_l1_error = 0.0;
        double total_l1_error = 0.0;
        const auto start = std::chrono::steady_clock::now();

        for (int sample = 0; sample < samples; ++sample) {
            fluxfp::Config config;
            config.theta = theta_distribution(generator);
            config.mean = mean_distribution(generator);
            config.sigma = sigma_distribution(generator);

            const double forecast_time = time_distribution(generator);
            const auto result = fluxfp::solve(forecast_time, config, cells);
            double mass = 0.0;
            double l1_error = 0.0;

            // Validate each example before saving it.
            for (int i = 0; i < cells; ++i) {
                const double density = result.density[i];

                if (!std::isfinite(density) || density < 0.0) {
                    throw std::runtime_error("Generated an invalid density.");
                }

                mass += density * result.dx;

                const double exact = fluxfp::exact_density(result.x[i], forecast_time, config);
                
                l1_error += std::abs(density - exact) * result.dx;
            }

            if (std::abs(mass - 1.0) > 1e-8) {
                throw std::runtime_error("Generated a distribution with invalid mass.");
            }

            largest_l1_error = std::max(largest_l1_error, l1_error);
            total_l1_error += l1_error;

            output
                << config.theta << ","
                << config.mean << ","
                << config.sigma << ","
                << forecast_time;

            for (double density : result.density) {
                output << "," << density;
            }

            output << "\n";

            if ((sample + 1) % 500 == 0) {
                std::cout
                    << "Generated "
                    << sample + 1
                    << " / "
                    << samples
                    << " examples\n";
            }
        }

        output.close();

        if (!output) {
            throw std::runtime_error("Failed while writing the dataset.");
        }

        const double elapsed_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

        std::ofstream metadata("data/dataset_metadata.json");

        if (!metadata) {
            throw std::runtime_error("Could not open the metadata file.");
        }

        metadata << std::setprecision(17);
        metadata
            << "{\n"
            << "  \"samples\": " << samples << ",\n"
            << "  \"cells\": " << cells << ",\n"
            << "  \"left\": -6.0,\n"
            << "  \"right\": 6.0,\n"
            << "  \"initial_mean\": 0.0,\n"
            << "  \"initial_std\": 0.3,\n"
            << "  \"seed\": " << seed << ",\n"
            << "  \"input_columns\": "
            << "[\"theta\", \"mean\", \"sigma\", \"time\"],\n"
            << "  \"parameter_ranges\": {\n"
            << "    \"theta\": [0.5, 2.0],\n"
            << "    \"mean\": [-1.0, 1.0],\n"
            << "    \"sigma\": [0.4, 1.0],\n"
            << "    \"time\": [0.1, 2.0]\n"
            << "  },\n"
            << "  \"generation_seconds\": "
            << elapsed_seconds << ",\n"
            << "  \"mean_solver_l1_error\": "
            << total_l1_error / samples << ",\n"
            << "  \"max_solver_l1_error\": "
            << largest_l1_error << "\n"
            << "}\n";

        metadata.close();

        if (!metadata) {
            throw std::runtime_error("Failed while writing metadata.");
        }

        std::cout
            << "\nSaved data/dataset.csv\n"
            << "Saved data/dataset_metadata.json\n"
            << "Generation time: "
            << elapsed_seconds << " seconds\n"
            << "Mean solver L1 error: "
            << total_l1_error / samples << "\n"
            << "Maximum solver L1 error: "
            << largest_l1_error << "\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "Error: "
            << error.what()
            << "\n";

        return 1;
    }
}