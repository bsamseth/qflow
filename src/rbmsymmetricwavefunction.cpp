#include "rbmsymmetricwavefunction.hpp"

#include <algorithm>
#include <cassert>
#include <vector>

RBMSymmetricWavefunction::RBMSymmetricWavefunction(int  M,
                                                   int  N,
                                                   int  f,
                                                   Real sigma2,
                                                   Real root_factor)
    : RBMWavefunction(M, N, sigma2, root_factor), _f(f)
{
    _parameters = RowVector::Zero(f + N + f * N);
    for (int i = 0; i < _parameters.size(); ++i)
    {
        _parameters[i] = rnorm_small_func();
    }
}

Real RBMSymmetricWavefunction::deriv_a(int k, const System& system) const
{
    Real res = 0;
    for (int i = k; i < _M; i += _f)
    {
        res += _root_factor * (system.data()[i] - _parameters[a(i)]) / _sigma2;
    }
    return res;
}

Real RBMSymmetricWavefunction::deriv_w(int k, int l, const System& system) const
{
    Real res = 0;
    for (int i = k; i < _M; i += _f)
    {
        res += _root_factor / (1 + std::exp(-v_j(l, system))) * system.data()[i]
               / _sigma2;
    }
    return res;
}

RowVector RBMSymmetricWavefunction::gradient(const System& system)
{
    RowVector grad_vec(_parameters.size());

    for (int i = 0; i < _f; ++i)
    {
        grad_vec[a(i)] = deriv_a(i, system);
    }
    for (int j = 0; j < _N; ++j)
    {
        grad_vec[b(j)] = deriv_b(j, system);
    }
    for (int i = 0; i < _f; ++i)
    {
        for (int j = 0; j < _N; ++j)
        {
            grad_vec[w(i, j)] = deriv_w(i, j, system);
        }
    }

    return grad_vec;
}
