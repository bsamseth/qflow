#include "optimizer.hpp"

#include "definitions.hpp"
#include "hamiltonian.hpp"
#include "sampler.hpp"
#include "simplegaussian.hpp"
#include "system.hpp"
#include "wavefunction.hpp"

#include <iostream>

SgdOptimizer::SgdOptimizer(Real eta) : _eta(eta) {}

RowVector SgdOptimizer::update_term(const RowVector& gradient)
{
    return -_eta * gradient;
}

AdamOptimizer::AdamOptimizer(std::size_t n_parameters,
                             Real        alpha,
                             Real        beta1,
                             Real        beta2,
                             Real        epsilon)
    : _alpha(alpha)
    , _beta1(beta1)
    , _beta2(beta2)
    , _epsilon(epsilon)
    , _t(0)
    , _m(RowVector::Zero(n_parameters))
    , _v(RowVector::Zero(n_parameters))
{
}

RowVector AdamOptimizer::update_term(const RowVector& gradient)
{
    ++_t;
    _m = _beta1 * _m + (1 - _beta1) * gradient;
    _v = _beta2 * _v + (1 - _beta2) * (gradient.cwiseProduct(gradient));
    _alpha_t
        = _alpha * std::sqrt(1 - std::pow(_beta2, _t)) / (1 - std::pow(_beta1, _t));

    RowVector update(_m.size());
    for (int i = 0; i < _m.size(); ++i)
    {
        update[i] = -_alpha_t * _m[i] / (std::sqrt(_v[i]) + _epsilon);
    }
    return update;
}
