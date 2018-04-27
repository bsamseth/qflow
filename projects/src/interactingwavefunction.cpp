#include <cmath>
#include <cassert>

#include "definitions.hpp"
#include "vector.hpp"
#include "system.hpp"
#include "interactingwavefunction.hpp"

InteractingWavefunction::InteractingWavefunction(std::initializer_list<Real> parameters)
    : SimpleGaussian(parameters)
{
    // Set default parameters.
    const static Vector defaults = std::vector<Real>{{0.5, 1, 0}};
    _parameters = defaults;

    // Copy any given parameters.
    int i = 0;
    for (auto it = parameters.begin(); it != parameters.end() and i < defaults.size(); ++it, ++i)
        _parameters[i] = *it;
}

Real InteractingWavefunction::correlation(System &system) const {
    const auto a = _parameters[2];
    Real f = 1;
    for (int i = 0; i < system.get_n_particles() - 1; ++i) {
        for (int j = i + 1; j < system.get_n_particles(); ++j) {

            Real r_ij = system.distance(i, j);

            if (r_ij <= a)
                return 0.0;

            f *= 1 - a / r_ij;
        }
    }
    return f;
}

Real InteractingWavefunction::operator() (System &system) const {
    return correlation(system) * SimpleGaussian::operator()(system);
}

Real InteractingWavefunction::laplacian(System &system) const {
    const auto alpha = _parameters[0];
    const auto beta  = _parameters[1];
    const auto a     = _parameters[2];
    const Real one_body_beta_term = - (system.get_dimensions() == 3 ? 2 + beta : system.get_dimensions());

    Real lap = 0;

    for (int k = 0; k < system.get_n_particles(); ++k) {

        const Vector &r_k = system[k];
        Vector r_k_hat = system[k];
        if (system.get_dimensions() == 3) {
            r_k_hat[2] *= beta;
        }

        lap += 2 * alpha * (2 * alpha * square(r_k_hat) + one_body_beta_term);

        // Interaction terms:

        Vector term1 (system.get_dimensions());
        Real term2 = 0;
        Real term3 = 0;
        for (int j = 0; j < system.get_n_particles(); ++j) {
            if (j == k) continue;
            const Vector r_kj = r_k - system[j];
            const Real r_kj_norm = system.distance(k, j);

            term1 += r_kj * (a / (square(r_kj_norm) * (r_kj_norm - a)));

            term3 += a * (a - 2*r_kj_norm) / (square(r_kj_norm) * square(r_kj_norm - a))
                + 2 * a / (square(r_kj_norm) * (r_kj_norm - a));

            for (int i = 0; i < system.get_n_particles(); ++i) {
                if (i == k) continue;
                const Vector r_ki = r_k - system[i];
                const Real r_ki_norm = system.distance(k, i);

                term2 += (r_ki * r_kj) * square(a) / (square(r_ki_norm * r_kj_norm) * (r_ki_norm - a) * (r_kj_norm - a));
            }
        }

        Real term1_final = -4 * alpha * (r_k_hat * term1);

        lap += term1_final + term2 + term3;

    }
    return lap;
}

