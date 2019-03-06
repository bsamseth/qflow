#include "definitions.hpp"

#include <random>

std::mt19937_64                      rand_gen(12345);
std::uniform_real_distribution<Real> unif(0, 1);
std::uniform_real_distribution<Real> centered(-0.5, 0.5);
std::normal_distribution<Real>       rnorm(0, 1);
std::normal_distribution<Real>       rnorm_small(0, 0.1);
