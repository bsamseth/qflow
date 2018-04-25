#include <cmath>

#include "definitions.hpp"
#include "vector.hpp"
#include "system.hpp"
#include "simplegaussian.hpp"

namespace {
    Real exponent(const System &system, Real beta) {
        Real g = 0;
        if (system.get_dimensions() == 3) {
            for (const Vector &boson : system.get_particles()) {
                g += square(boson[0]) + square(boson[1]) + beta * square(boson[2]);
            }
        } else {
            for (const Vector &boson : system.get_particles()) {
                g += square(boson);
            }
        }
        return g;
    }
}

SimpleGaussian::SimpleGaussian(std::initializer_list<Real> parameters)
    : Wavefunction(parameters)
{
    if (_parameters.size() > 0)
        _alpha = _parameters[0];
    if (_parameters.size() > 1)
        _beta = _parameters[1];
    _parameters[0] = _alpha;
    _parameters[1] = _beta;
}
Real SimpleGaussian::operator() (System &system) const {
    return std::exp( - _alpha * exponent(system, _beta) );
}

Real SimpleGaussian::derivative_alpha(const System &system) const {
    return - exponent(system, _beta);
}

Vector SimpleGaussian::gradient(System &system) const {
    Vector grad(1);
    grad[0] = -exponent(system, _beta);
    return grad;
}

Real SimpleGaussian::laplacian(System &system) const {
    const Real one_body_beta_term = - (system.get_dimensions() == 3 ? 2 + _beta : system.get_dimensions());

    Real E_L = 0;

    for (int k = 0; k < system.get_n_particles(); ++k) {

        Vector r_k = system[k];
        if (system.get_dimensions() == 3) {
            r_k[2] *= _beta;
        }

        E_L += 2 * _alpha * (2 * _alpha * (r_k * r_k) + one_body_beta_term);
    }

    return E_L;
}
