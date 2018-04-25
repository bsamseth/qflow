#include "definitions.hpp"
#include "system.hpp"
#include "harmonicoscillatorhamiltonian.hpp"

Real HarmonicOscillatorHamiltonian::local_energy(System &system, const Wavefunction &psi) const {
    const Real alpha = psi.get_parameters()[0];
    const Real beta = psi.get_parameters()[1];
    const Real one_body_beta_term = - (system.get_dimensions() == 3 ? 2 + beta : system.get_dimensions());

    Real E_L = 0;

    for (int k = 0; k < system.get_n_particles(); ++k) {

        Vector r_k = system[k];
        if (system.get_dimensions() == 3) {
            r_k[2] *= beta;
        }

        E_L += 2 * alpha * (2 * alpha * (r_k * r_k) + one_body_beta_term);
    }

    return - 0.5 * E_L + external_potential(system) + internal_potential(system);
}

Real HarmonicOscillatorHamiltonian::external_potential(System &system) const {
    Real pot = 0;

    for (int i = 0; i < system.get_n_particles(); ++i) {
        if (system.get_dimensions() == 3) {
            pot += square(system[i][0]) + square(system[i][1]) + square(_omega_z * system[i][2]);
        } else {
            pot += system[i] * system[i];
        }
    }

    return 0.5 * pot;
}

