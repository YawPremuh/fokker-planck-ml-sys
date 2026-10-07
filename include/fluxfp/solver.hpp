#pragma once

namespace fluxfp {

struct Config {
    double theta = 1.0;
    double mean = 1.0;
    double sigma = 0.8;
    double initial_mean = 0.0;
    double initial_std = 0.3;
};

double exact_density(double x, double time, const Config& config);

}