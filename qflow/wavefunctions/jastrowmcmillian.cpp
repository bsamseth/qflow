#include "jastrowmcmillian.hpp"

#include "distance.hpp"

#include <cmath>

JastrowMcMillian::JastrowMcMillian(int n, Real beta)
    : Wavefunction(vector_from_sequence({beta})), n_(n)
{
    assert(_parameters[0] == beta);
    assert(_parameters.size() == 1);
}

Real JastrowMcMillian::operator()(const System& system)
{
    const Real beta = _parameters[0];
    const Real L    = Distance::get_simulation_box_size();
    if (L < 14)
        throw 123;

    const int N   = system.rows();
    Real      res = 0;
    for (int i = 0; i < N - 1; ++i)
    {
        for (int j = i + 1; j < N; ++j)
        {
            const Real r_ij = Distance::probe(system, i, j);
            res += 2 * r_ij < L ? std::pow(beta / r_ij, n_) : 0;
        }
    }
    return std::exp(-0.5 * res);
}

RowVector JastrowMcMillian::gradient(const System& system)
{
    const Real beta = _parameters[0];
    RowVector  grad = RowVector::Zero(1);
    const int  N    = system.rows();
    for (int i = 0; i < N - 1; ++i)
    {
        for (int j = i + 1; j < N; ++j)
        {
            const Real r_ij = Distance::probe(system, i, j);
            grad[0] += std::pow(1 / r_ij, n_);
        }
    }
    grad[0] *= -0.5 * n_ * std::pow(beta, n_ - 1);
    return grad;
}

Real JastrowMcMillian::drift_force(const System& system, int k, int d)
{
    const Real beta = _parameters[0];
    const int  N    = system.rows();
    Real       res  = 0;
    for (int i = 0; i < N; ++i)
    {
        if (i != k)
        {
            const Real r_ik = Distance::probe(system, i, k);
            res += std::pow(beta / r_ik, n_) * (system(k, d) - system(i, d))
                   / (r_ik * r_ik);
        }
    }
    return 2 * 0.5 * n_ * res;
}

Real JastrowMcMillian::laplacian(const System& system)
{
    const Real beta = _parameters[0];
    const int  N    = system.rows();
    const int  D    = system.cols();

    Real res = 0;
    for (int k = 0; k < N; ++k)
    {
        RowVector term1 = RowVector::Zero(D);
        Real      term2 = 0;

        for (int i = 0; i < N; ++i)
        {
            if (i != k)
            {
                const Real r_ik = Distance::probe(system, i, k);
                const auto diff = system.row(k) - system.row(i);

                term1 += std::pow(beta / r_ik, n_) * diff / (r_ik * r_ik);

                term2 += std::pow(beta / r_ik, n_)
                         * (D - diff.squaredNorm() * (n_ + 2) / (r_ik * r_ik))
                         / (r_ik * r_ik);
            }
        }
        res += 0.25 * n_ * n_ * term1.squaredNorm() + n_ * 0.5 * term2;
    }

    return res;
}
