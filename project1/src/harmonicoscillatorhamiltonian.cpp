#include "definitions.hpp"
#include "system.hpp"
#include "harmonicoscillatorhamiltonian.hpp"

Real HarmonicOscillatorHamiltonian::local_energy(const System &system, const Wavefunction &psi) const {
    const Real alpha = psi.get_alpha();
    const Real beta = psi.get_beta();
    const Real one_body_beta_term = - (system.get_dimensions() == 3 ? 2 + beta : system.get_dimensions());

    Real E_L = 0;

    for (int k = 0; k < system.get_n_bosons(); ++k) {

        Boson r_k = system[k];
        if (system.get_dimensions() == 3) {
            r_k[2] *= beta;
        }

        E_L += 2 * alpha * (2 * alpha * (r_k * r_k) + one_body_beta_term);
    }

    return - 0.5 * E_L + external_potential(system) + internal_potential(system);
}

Real HarmonicOscillatorHamiltonian::external_potential(const System &system) const {
    Real pot = 0;

    for (int i = 0; i < system.get_n_bosons(); ++i) {
        if (system.get_dimensions() == 3) {
            pot += square(system[i][0]) + square(system[i][1]) + square(_omega_z * system[i][2]);
        } else {
            pot += system[i] * system[i];
        }
    }

    return 0.5 * pot;
}

Real HarmonicOscillatorHamiltonian::derivative_alpha(const System &system, const Wavefunction &psi) const {
    const Real alpha = psi.get_alpha();
    const Real beta = psi.get_beta();
    const int dim = system.get_dimensions();
    const int N = system.get_n_bosons();
    const Real one_body_beta_term = (dim == 3 ? 2 + beta : dim);

    Real sum_squared_dist = 0;
    /* if (dim == 3) { */
    /*     for (Boson boson : system.get_bosons()) { */
    /*         boson[2] *= beta; */
    /*         sum_squared_dist += square(boson); */
    /*     } */
    /* } else { */
        for (const Boson &boson : system.get_bosons()) {
            sum_squared_dist += square(boson);
        }
    /* } */

    // Add contribution by beta if any.
    if (dim == 3 and beta != 1) {
        for (const Boson &boson : system.get_bosons()) {
            sum_squared_dist += square(boson[2]) * (square(beta) - 1);
        }
    }

    return N * one_body_beta_term - 4 * alpha * sum_squared_dist;
}
