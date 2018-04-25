#include <limits>
#include <cassert>
#include <cmath>

#include "definitions.hpp"
#include "system.hpp"
#include "interactinghamiltonian.hpp"

Real InteractingHamiltonian::local_energy(System &system, const Wavefunction &psi) const {
    // If internal potential is active ( > 0 means inf), no need to check more.
    if (internal_potential(system) > 0)
        return std::numeric_limits<Real>::max();

    const Real alpha = psi.get_alpha();
    const Real beta = psi.get_beta();
    assert(_a == psi.get_a());
    const Real one_body_beta_term = - (system.get_dimensions() == 3 ? 2 + beta : system.get_dimensions());

    Real E_L = 0;

    for (int k = 0; k < system.get_n_particles(); ++k) {

        const Vector &r_k = system[k];
        Vector r_k_hat = system[k];
        if (system.get_dimensions() == 3) {
            r_k_hat[2] *= beta;
        }

        E_L += 2 * alpha * (2 * alpha * square(r_k_hat) + one_body_beta_term);

        // Interaction terms:

        Vector term1 (system.get_dimensions());
        Real term2 = 0;
        Real term3 = 0;
        for (int j = 0; j < system.get_n_particles(); ++j) {
            if (j == k) continue;
            const Vector r_kj = r_k - system[j];
            const Real r_kj_norm = system.distance(k, j);

            term1 += r_kj * (_a / (square(r_kj_norm) * (r_kj_norm - _a)));

            term3 += _a * (_a - 2*r_kj_norm) / (square(r_kj_norm) * square(r_kj_norm - _a))
                + 2 * _a / (square(r_kj_norm) * (r_kj_norm - _a));

            for (int i = 0; i < system.get_n_particles(); ++i) {
                if (i == k) continue;
                const Vector r_ki = r_k - system[i];
                const Real r_ki_norm = system.distance(k, i);

                term2 += (r_ki * r_kj) * square(_a) / (square(r_ki_norm * r_kj_norm) * (r_ki_norm - _a) * (r_kj_norm - _a));
            }
        }

        Real term1_final = -4 * alpha * (r_k_hat * term1);

        E_L += term1_final + term2 + term3;

    }
    return - 0.5 * E_L + external_potential(system);
}

Real InteractingHamiltonian::internal_potential(System &system) const {

    for (int i = 0; i < system.get_n_particles() - 1; ++i) {
        for (int j = i + 1; j < system.get_n_particles(); ++j) {
            if (system.distance(i, j) <= _a)
                return std::numeric_limits<Real>::max();
        }
    }

    return 0.0;
}
