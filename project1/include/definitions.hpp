#pragma once
#include <random>

using Real = double;

template<typename T>
inline auto square(T x) {
    return x * x;
}

extern std::mt19937_64 rand_gen;
extern std::uniform_real_distribution<Real> unif;
extern std::uniform_real_distribution<Real> centered;
extern std::normal_distribution<Real> rnorm;
