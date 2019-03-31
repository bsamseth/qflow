#include "jastrowpade.hpp"

#include "distance.hpp"

#include <cmath>

JastrowPade::JastrowPade(Real alpha, Real beta)
    : Wavefunction(vector_from_sequence({beta})), alpha_(alpha)
{
    assert(_parameters[0] == beta);
    assert(_parameters.size() == 1);
}

Real JastrowPade::operator()(const System& system)
{
    const Real beta = _parameters[0];
    const int  N    = system.rows();
    Real       res  = 0;
    for (int i = 0; i < N - 1; ++i)
    {
        for (int j = i + 1; j < N; ++j)
        {
            const Real r_ij = Distance::probe(system, i, j);
            res += alpha_ * r_ij / (1 + beta * r_ij);
        }
    }
    return std::exp(res);
}

RowVector JastrowPade::gradient(const System& system)
{
    const Real beta = _parameters[0];
    RowVector  grad = RowVector::Zero(1);
    const int  N    = system.rows();
    for (int i = 0; i < N - 1; ++i)
    {
        for (int j = i + 1; j < N; ++j)
        {
            const Real r_ij = Distance::probe(system, i, j);
            grad[0] -= alpha_ * r_ij * r_ij / square(1 + beta * r_ij);
        }
    }
    return grad;
}

Real JastrowPade::drift_force(const System& system, int k, int d)
{
    const Real beta = _parameters[0];
    const int  N    = system.rows();
    Real       res  = 0;
    for (int i = 0; i < N; ++i)
    {
        if (i != k)
        {
            const Real r_ik = Distance::probe(system, i, k);
            res += alpha_ * (system(k, d) - system(i, d))
                   / (square(1 + beta * r_ik) * r_ik);
        }
    }
    return 2 * res;
}

Real JastrowPade::laplacian(const System& system)
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
                const Real r_ik       = Distance::probe(system, i, k);
                const Real beta_rik_1 = 1 + beta * r_ik;
                const auto diff       = system.row(k) - system.row(i);

                term1 += diff / (square(beta_rik_1) * r_ik);

                term2 -= diff.squaredNorm() * (3 * beta * r_ik + 1)
                         / (square(beta_rik_1) * beta_rik_1 * square(r_ik) * r_ik);
                term2 += D / (square(beta_rik_1) * r_ik);
            }
        }
        res += alpha_ * alpha_ * term1.squaredNorm() + alpha_ * term2;
    }

    return res;
}
