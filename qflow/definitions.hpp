#pragma once
#include <Eigen/Dense>
#include <random>

// Macro for silencing  unused parameters on demand.
#define SUPPRESS_WARNING(a) (void) a

/**
 * Defining an alias for double. Used so that if we ever want to change
 * this to float (for improved speed, reduced accuracy), this is done
 * one place, as apposed to a project wide refactor.
 * \typedef
 */
using Real = double;

constexpr Real PI = 3.14159265358979323846; /**< Circle constant. */

constexpr Real NUMMERIC_DIFF_STEP
    = 0.001; /**< Default step used in numerical differentiation. */

/**
 * Define various linear algebra types. Note that row-major storage order
 * is used. Eigen should be equally performing with both, but row-major is
 * compatible with NumPy and hence bindings are easy to make.
 */
using Matrix    = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Array     = Eigen::Array<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Vector    = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
using RowVector = Eigen::Matrix<Real, 1, Eigen::Dynamic>;

/**
 * Corresponding Eigen::Ref bindings for each linear algebra type.
 * Most computations can be used as one of these types, useful for writing
 * generic functions without having to use templates (which can make it
 * difficult writing bindings for).
 */
using MatrixRef    = Eigen::Ref<Matrix>;
using ArrayRef     = Eigen::Ref<Array>;
using VectorRef    = Eigen::Ref<Vector>;
using RowVectorRef = Eigen::Ref<RowVector>;

/**
 * Generic template for computing x * x, for any type. Useful
 * when x is some expression which we do not want to compute twice.
 * @param x Value to multiply with itself.
 * @return Result of `x * x`.
 */
template <typename T>
constexpr auto square(T&& x)
{
    return x * x;
}
template <typename T>
constexpr auto square(const T& x)
{
    return x * x;
}

/** Type safe sign(x). */
template <typename T>
inline constexpr int sign(T x, std::false_type)
{
    return T(0) < x;
}
template <typename T>
inline constexpr int sign(T x, std::true_type)
{
    return (T(0) < x) - (x < T(0));
}
template <typename T>
inline constexpr int sign(T x)
{
    return sign(x, std::is_signed<T>());
}

// Random number generation.
extern std::mt19937_64 rand_gen; /**< Random number generator used. */
extern std::uniform_real_distribution<Real>
    unif; /**< Uniform random distribution (0, 1). */
extern std::uniform_real_distribution<Real>
                                      centered; /**< Uniform random distribution (-.5, .5). */
extern std::normal_distribution<Real> rnorm; /**< N(0, 1) random number distribution. */
extern std::normal_distribution<Real>
    rnorm_small; /**< N(0, 0.1) random number distribution. */

inline auto unif_func()
{
    return unif(rand_gen);
}
inline auto centered_func()
{
    return centered(rand_gen);
}
inline auto rnorm_func()
{
    return rnorm(rand_gen);
}
inline auto rnorm_small_func()
{
    return rnorm_small(rand_gen);
}
