#include <limits>
#include <cassert>
#include <cmath>

#include "definitions.hpp"
#include "system.hpp"
#include "interactinghamiltonian.hpp"

Real InteractingHamiltonian::local_energy(const System &system, const Wavefunction &psi) const {
    const Real alpha = psi.get_alpha();
    const Real beta = psi.get_beta();
    assert(_a == psi.get_a());
    const Real one_body_beta_term = - (system.get_dimensions() == 3 ? 2 + beta : system.get_dimensions());

    Real E_L = 0;

    for (int k = 0; k < system.get_n_bosons(); ++k) {

        const Boson &r_k = system[k];
        Boson r_k_hat = system[k];
        if (system.get_dimensions() == 3) {
            r_k_hat[2] *= beta;
        }

        E_L += 2 * alpha * (2 * alpha * square(r_k_hat) + one_body_beta_term);

        // Interaction terms:

        Boson term1 (system.get_dimensions());
        Real term2 = 0;
        Real term3 = 0;
        for (int j = 0; j < system.get_n_bosons(); ++j) {
            if (j == k) continue;
            const Boson r_kj = r_k - system[j];
            const Real r_kj_norm = std::sqrt( square( r_kj ) );

            term1 += r_kj * (_a / (square(r_kj_norm) * (r_kj_norm - _a)));

            term3 += _a * (_a - 2*r_kj_norm) / (square(r_kj_norm) * square(r_kj_norm - _a))
                + 2 * _a / (square(r_kj_norm) * (r_kj_norm - _a));

            for (int i = 0; i < system.get_n_bosons(); ++i) {
                if (i == k) continue;
                const Boson r_ki = r_k - system[i];
                const Real r_ki_norm = std::sqrt( square( r_ki ) );

                term2 += (r_ki * r_kj) * square(_a) / (square(r_ki_norm * r_kj_norm) * (r_ki_norm - _a) * (r_kj_norm - _a));
            }
        }

        Real term1_final = -4 * alpha * (r_k_hat * term1);

        E_L += term1_final + term2 + term3;

    }

    if (internal_potential(system) > 0)
        return std::numeric_limits<Real>::max();

    return - 0.5 * E_L + external_potential(system);
}

Real InteractingHamiltonian::internal_potential(const System &system) const {

    for (int i = 0; i < system.get_n_bosons() - 1; ++i) {
        const Boson &r_i = system[i];
        for (int j = i + 1; j < system.get_n_bosons(); ++j) {
            if (square(r_i - system[j]) <= square(_a))
                return std::numeric_limits<Real>::max();
        }
    }

    return 0.0;
}

Real InteractingHamiltonian::derivative_alpha(const System &system, const Wavefunction &psi) const {

    Real mean_interaction_contribution = 0;

    for (int k = 0; k < system.get_n_bosons(); ++k) {
        const Boson &r_k = system[k];
        Boson r_k_hat = system[k];
        if (system.get_dimensions() == 3) {
            r_k_hat[2] *= psi.get_beta();
        }

        Boson term1 (system.get_dimensions());
        for (int j = 0; j < system.get_n_bosons(); ++j) {
            if (j == k) continue;
            const Boson r_kj = r_k - system[j];
            const Real r_kj_norm = std::sqrt( square( r_kj ) );

            term1 += r_kj * (_a / (square(r_kj_norm) * (r_kj_norm - _a)));
        }

        mean_interaction_contribution += -2 * psi.get_alpha() * (r_k_hat * term1);
    }

    return HarmonicOscillatorHamiltonian::derivative_alpha(system, psi) + mean_interaction_contribution;
}

