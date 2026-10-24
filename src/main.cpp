#include "fluxfp/solver.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

int main() {
    try {
        fluxfp::Config config;

        const double forecast_time = 1.0;
        const double left = -6.0;
        const double right = 6.0;
        const int cells = 400;
        const double dx = (right - left) / cells;
        const auto result = fluxfp::solve(forecast_time, config, cells);

        std::filesystem::create_directories("results");
        std::ofstream output("results/comparison.csv");

        if (!output) {
            throw std::runtime_error("Runtime Error: Could not open the output file.");
        }
        output << std::setprecision(17);
        output << "x,numerical_density,analytical_density\n";
        double total_probability = 0.0;
        double l1_error = 0.0;

        for (int i = 0; i < cells; ++i) {
            // Calculate the known answer at the same position.
            const double exact =
                fluxfp::exact_density(result.x[i], forecast_time, config);

            // Save both answers side by side.
            output
                << result.x[i] << ","
                << result.density[i] << ","
                << exact << "\n";

            total_probability += result.density[i] * result.dx;

            l1_error += std::abs(result.density[i] - exact) * result.dx;
        }

        output.close();

        if (!output) {
            throw std::runtime_error("An error occurred while writing the file.");
        }

        const double minimum_density = *std::min_element(result.density.begin(), result.density.end());

        std::cout << std::setprecision(10);

        std::cout
            << "Forecast time: " << forecast_time << "\n"
            << "Cells: " << cells << "\n"
            << "Time steps: " << result.steps << "\n"
            << "Total probability: "
            << total_probability << "\n"
            << "Minimum density: "
            << minimum_density << "\n"
            << "L1 error: "
            << l1_error << "\n"
            << "Saved results/comparison.csv\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";

        return 1;
    }
}