#pragma once

#include <vector>

namespace fluxfp {

struct Config {
    double theta = 1.0;
    double mean = 1.0;
    double sigma = 0.8;
    double initial_mean = 0.0;
    double initial_std = 0.3;
};

struct Result {
    std::vector<double> x;
    std::vector<double> density;
    double dx = 0.0;
    int steps = 0;
};

double exact_density(double x, double time, const Config& config);
Result solve(double time, const Config& config, int cells = 400);

}