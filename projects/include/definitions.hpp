#pragma once
#include <random>

// Macro for silencing  unused parameters on demand.
#define SUPPRESS_WARNING(a) (void)a

/**
 * Defining an alias for double. Used so that if we ever want to change
 * this to float (for improved speed, reduced accuracy), this is done
 * one place, as apposed to a project wide refactor.
 * \typedef
 */
using Real = double;

constexpr Real PI = 3.14159265358979323846; /**< Circle constant. */

/**
 * Generic template for computing x * x, for any type.
 * @param x Value to multiply with itself.
 * @return Result of `x * x`.
 */
template<typename T>
inline auto square(T x) {
    return x * x;
}

extern std::mt19937_64 rand_gen;  /**< Random number generator used. */
extern std::uniform_real_distribution<Real> unif;  /**< Uniform random distribution (0, 1). */
extern std::uniform_real_distribution<Real> centered;  /**< Uniform random distribution (-.5, .5). */
extern std::normal_distribution<Real> rnorm;  /**< N(0, 1) random number distribution. */
extern std::normal_distribution<Real> rnorm_small;  /**< N(0, 0.1) random number distribution. */

inline auto unif_func() {
    return unif(rand_gen);
}
inline auto centered_func() {
    return centered(rand_gen);
}
inline auto rnorm_func() {
    return rnorm(rand_gen);
}
inline auto rnorm_small_func() {
    return rnorm_small(rand_gen);
}
