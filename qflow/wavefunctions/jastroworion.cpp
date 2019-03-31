#include "jastroworion.hpp"

#include "distance.hpp"

#include <cmath>

JastrowOrion::JastrowOrion(Real beta, Real gamma)
    : Wavefunction(vector_from_sequence({beta, gamma}))
{
}

Real JastrowOrion::operator()(const System& system)
{
    const Real beta = _parameters[0], gamma = _parameters[1];
    const int  N   = system.rows();
    Real       res = 0;
    for (int i = 0; i < N - 1; ++i)
    {
        for (int j = i + 1; j < N; ++j)
        {
            const Real r_ij = Distance::probe(system, i, j);
            res += -0.5 * square(beta * r_ij) + std::abs(beta * gamma) * r_ij;
        }
    }
    return std::exp(res);
}

RowVector JastrowOrion::gradient(const System& system)
{
    const Real beta = _parameters[0], gamma = _parameters[1];
    RowVector  grad = RowVector::Zero(2);
    const int  N    = system.rows();
    for (int i = 0; i < N - 1; ++i)
    {
        for (int j = i + 1; j < N; ++j)
        {
            const Real r_ij = Distance::probe(system, i, j);
            grad[0] += r_ij * (-beta * r_ij + gamma * sign(gamma * beta));
            grad[1] += r_ij;
        }
    }
    grad[1] *= beta * sign(beta * gamma);
    return grad;
}

Real JastrowOrion::drift_force(const System& system, int k, int d)
{
    const Real beta = _parameters[0], gamma = _parameters[1];
    const int  N   = system.rows();
    Real       res = 0;
    for (int i = 0; i < k; ++i)
    {
        const Real r_ik = Distance::probe(system, i, k);
        res += (-square(beta) + std::abs(beta * gamma) / r_ik)
               * (system(k, d) - system(i, d));
    }
    for (int j = k + 1; j < N; ++j)
    {
        const Real r_kj = Distance::probe(system, k, j);
        res += (-square(beta) + std::abs(beta * gamma) / r_kj)
               * (system(k, d) - system(j, d));
    }
    return 2 * res;
}

Real JastrowOrion::laplacian(const System& system)
{
    const Real beta = _parameters[0], gamma = _parameters[1];
    const int  N = system.rows();
    const int  D = system.cols();

    Real res = 0;
    for (int k = 0; k < N; ++k)
    {
        RowVector term1 = RowVector::Zero(D);
        Real      term2 = 0;

        for (int i = 0; i < k; ++i)
        {
            const Real r_ik = Distance::probe(system, i, k);
            term1 += (-square(beta) + std::abs(beta * gamma) / r_ik)
                     * (system.row(k) - system.row(i));
            term2 += (-square(beta) + std::abs(beta * gamma) / r_ik) * D;
            term2 -= (system.row(i) - system.row(k)).squaredNorm()
                     * std::abs(beta * gamma) / (r_ik * r_ik * r_ik);
        }
        for (int j = k + 1; j < N; ++j)
        {
            const Real r_kj = Distance::probe(system, k, j);
            term1 += (-square(beta) + std::abs(beta * gamma) / r_kj)
                     * (system.row(k) - system.row(j));
            term2 += (-square(beta) + std::abs(beta * gamma) / r_kj) * D;
            term2 -= (system.row(j) - system.row(k)).squaredNorm()
                     * std::abs(beta * gamma) / (r_kj * r_kj * r_kj);
        }
        res += term1.squaredNorm() + term2;
    }

    return res;
}
