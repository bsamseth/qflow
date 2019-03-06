#include "simplegaussian.hpp"

#include "definitions.hpp"
#include "system.hpp"
#include "vector.hpp"

#include <cassert>
#include <cmath>

namespace
{
Real exponent(const System& system, Real beta)
{
    const bool is_3D = system.cols() == 3;
    Real       g     = 0;
    for (int p = 0; p < system.rows(); ++p)
    {
        const auto& boson = system.row(p);
        if (is_3D)
        {
            g += square(boson[0]) + square(boson[1]) + beta * square(boson[2]);
        }
        else
        {
            g += squaredNorm(boson);
        }
    }
    return g;
}
}  // namespace

SimpleGaussian::SimpleGaussian(Real alpha, Real beta)
    : SimpleGaussian(vector_from_sequence({alpha, beta}))
{
}

/* SimpleGaussian::SimpleGaussian(const RowVector& parameters) */
/*     : Wavefunction(parameters) */
/* {} */

Real SimpleGaussian::operator()(const System& system)
{
    const auto alpha = _parameters[0];
    const auto beta  = _parameters[1];
    return std::exp(-alpha * exponent(system, beta));
}

Real SimpleGaussian::derivative_alpha(const System& system) const
{
    const auto beta = _parameters[1];
    return -exponent(system, beta);
}

RowVector SimpleGaussian::gradient(const System& system)
{
    const auto beta = _parameters[1];
    RowVector  grad(2);
    grad[0] = -exponent(system, beta);
    grad[1] = 0;  // Fix beta, alpha is only variational parameter.
    return grad;
}

Real SimpleGaussian::laplacian(const System& system)
{
    const auto alpha              = _parameters[0];
    const auto beta               = _parameters[1];
    const Real one_body_beta_term = -(system.cols() == 3 ? 2 + beta : system.cols());

    Real E_L = 0;

    for (int k = 0; k < system.rows(); ++k)
    {
        RowVector r_k = system.row(k);
        if (system.cols() == 3)
        {
            r_k[2] *= beta;
        }

        E_L += 2 * alpha * (2 * alpha * (r_k.dot(r_k)) + one_body_beta_term);
    }
    return E_L;
}

void SimpleGaussian::set_parameters(const RowVector& parameters)
{
    _parameters[0] = parameters[0];
    // Fix beta, alpha is only variational parameter.
}
