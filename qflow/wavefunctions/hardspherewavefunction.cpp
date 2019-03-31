#include "hardspherewavefunction.hpp"

#include "definitions.hpp"
#include "distance.hpp"
#include "system.hpp"
#include "vector.hpp"

#include <cassert>
#include <cmath>

HardSphereWavefunction::HardSphereWavefunction(Real alpha, Real beta, Real a)
    : SimpleGaussian(vector_from_sequence({alpha, beta, a}))
{
}

Real HardSphereWavefunction::correlation(const System& system) const
{
    const auto a = _parameters[2];
    Real       f = 1;
    for (int i = 0; i < system.rows() - 1; ++i)
    {
        for (int j = i + 1; j < system.rows(); ++j)
        {
            Real r_ij = Distance::probe(system, i, j);

            if (r_ij <= a)
                return 0.0;

            f *= 1 - a / r_ij;
        }
    }
    return f;
}

Real HardSphereWavefunction::operator()(const System& system)
{
    return correlation(system) * SimpleGaussian::operator()(system);
}

Real HardSphereWavefunction::laplacian(const System& system)
{
    const auto alpha              = _parameters[0];
    const auto beta               = _parameters[1];
    const auto a                  = _parameters[2];
    const Real one_body_beta_term = -(system.cols() == 3 ? 2 + beta : system.cols());

    Real lap = 0;

    for (int k = 0; k < system.rows(); ++k)
    {
        const RowVector& r_k     = system.row(k);
        RowVector        r_k_hat = system.row(k);
        if (system.cols() == 3)
        {
            r_k_hat[2] *= beta;
        }

        lap += 2 * alpha * (2 * alpha * squaredNorm(r_k_hat) + one_body_beta_term);

        // Interaction terms:

        RowVector term1 = RowVector::Zero(system.cols());
        Real      term2 = 0;
        Real      term3 = 0;
        for (int j = 0; j < system.rows(); ++j)
        {
            if (j == k)
                continue;
            const RowVector r_kj      = r_k - system.row(j);
            const Real      r_kj_norm = Distance::probe(system, k, j);

            term1 += r_kj * (a / (square(r_kj_norm) * (r_kj_norm - a)));

            term3
                += a * (a - 2 * r_kj_norm) / (square(r_kj_norm) * square(r_kj_norm - a))
                   + 2 * a / (square(r_kj_norm) * (r_kj_norm - a));

            for (int i = 0; i < system.rows(); ++i)
            {
                if (i == k)
                    continue;
                const RowVector r_ki      = r_k - system.row(i);
                const Real      r_ki_norm = Distance::probe(system, k, i);

                term2 += (r_ki.dot(r_kj)) * square(a)
                         / (square(r_ki_norm * r_kj_norm) * (r_ki_norm - a)
                            * (r_kj_norm - a));
            }
        }

        Real term1_final = -4 * alpha * (r_k_hat.dot(term1));

        lap += term1_final + term2 + term3;
    }
    return lap;
}
