#include "jastrowpade.hpp"

#include "distance.hpp"

#include <cmath>

JastrowPade::JastrowPade(Real alpha, Real beta, bool alpha_constant)
  : Wavefunction(vector_from_sequence({alpha, beta})), alpha_is_constant(alpha_constant)
{
    assert(_parameters.size() == 2);
    assert(_parameters[0] == alpha);
    assert(_parameters[1] == beta);
}

Real JastrowPade::operator()(const System& system)
{
    const Real alpha = _parameters[0];
    const Real beta  = _parameters[1];
    const int  N    = system.rows();
    Real       res  = 0;
    for (int i = 0; i < N - 1; ++i)
    {
        for (int j = i + 1; j < N; ++j)
        {
            const Real r_ij = Distance::probe(system, i, j);
            res += alpha * r_ij / (1 + beta * r_ij);
        }
    }
    return std::exp(res);
}

RowVector JastrowPade::gradient(const System& system)
{
  const Real alpha = _parameters[0];
    const Real beta = _parameters[1];
    RowVector  grad = RowVector::Zero(2);
    const int  N    = system.rows();
    for (int i = 0; i < N - 1; ++i)
    {
        for (int j = i + 1; j < N; ++j)
        {
            const Real r_ij = Distance::probe(system, i, j);
            grad[0] += alpha_is_constant ? 0 : r_ij / (1 + beta * r_ij);
            grad[1] -= alpha * r_ij * r_ij / square(1 + beta * r_ij);
        }
    }
    return grad;
}

Real JastrowPade::drift_force(const System& system, int k, int d)
{
    const Real alpha = _parameters[0];
    const Real beta = _parameters[1];
    const int  N    = system.rows();
    Real       res  = 0;
    for (int i = 0; i < N; ++i)
    {
        if (i != k)
        {
            const Real r_ik = Distance::probe(system, i, k);
            res += alpha * (system(k, d) - system(i, d))
                   / (square(1 + beta * r_ik) * r_ik);
        }
    }
    return 2 * res;
}

Real JastrowPade::laplacian(const System& system)
{
  const Real alpha = _parameters[0];
    const Real beta = _parameters[1];
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
        res += alpha * alpha * term1.squaredNorm() + alpha * term2;
    }

    return res;
}
