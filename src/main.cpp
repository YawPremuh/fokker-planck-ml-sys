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

        std::filesystem::create_directories("results");
        std::ofstream output("results/analytical.csv");

        if (!output) {
            throw std::runtime_error("Runtime Error: Could not open the output file.");
        }
        output << std::setprecision(17);
        output << "x,density\n";
        double total_probability = 0.0;

        for (int i = 0; i < cells; ++i) {
            const double x = left + (i + 0.5) * dx;
            const double density = fluxfp::exact_density(x, forecast_time, config);
            output << x << "," << density << "\n";
            total_probability += density * dx;
        }

        output.close();

        if (!output) {
            throw std::runtime_error("An error occurred while writing the file.");
        }

        std::cout << std::setprecision(10);
        std::cout << "Forecast time: " << forecast_time << "\n";
        std::cout << "Approximate total probability: " << total_probability << "\n";
        std::cout << "Saved results/analytical.csv\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";

        return 1;
    }
}