#include <random>
#include <cassert>
#include <omp.h>

#include "vmcimportancesolver.hpp"

namespace VMC {

VMCImportanceSolver::VMCImportanceSolver(const VMCConfiguration &config) : VMCSolver(config) {}

void VMCImportanceSolver::quantum_force(const arma::Mat<Real> &R, arma::Col<Real> &Q_force, int particle) {

    int k = particle;  // Shorthand within function.

    const auto r_k = R.col(k);
    arma::Col<Real> r_k_skewed = R.col(k);
    if (_config.dims == Dimensions::DIM_3)
        r_k_skewed(2) *= _beta;

    Q_force = -2 * _alpha * r_k_skewed;

    // Other terms are only relevant with interaction enabled.
    if (_config.interaction == InteractionType::ON) {

        for (int j = 0; j < _config.n_particles; ++j) {
            if (j == k) continue;

            const auto r_kj = r_k - R.col(j);
            const Real r_kj_norm = dist(std::min(k, j), std::max(k, j));

            Q_force += r_kj * _config.a / (r_kj_norm * r_kj_norm * (r_kj_norm - _config.a));
        }
    }

    Q_force *= 2;
}

Results VMCImportanceSolver::run_MC(const int n_cycles, std::ostream *out, const double alpha, const double beta) {
    arma::Mat<Real> R_old (_config.dims, _config.n_particles);
    arma::Mat<Real> R_new (_config.dims, _config.n_particles);
    arma::Col<Real> Q_force_old (_config.dims);
    arma::Col<Real> Q_force_new (_config.dims);
    Real E_sum = 0, E2_sum = 0;

    this->_alpha = alpha;
    this->_beta = beta;

    // Random initial starting point.
    for (int i = 0; i < _config.n_particles; ++i) {
        for (int d = 0; d < _config.dims; ++d) {
            R_old(d, i) = R_new(d, i) = rnorm(rand_gen) * std::sqrt(_config.time_step);
        }
    }

    initialize_distance_matrix(R_old);

    int accepted_moves = 0;
    for (int cycle = 1; cycle <= n_cycles; ++cycle) {

        Real Psi_old = Psi(R_old);

        for (int i = 0; i < _config.n_particles; ++i) {

            // Calculate quantum force on particle i for use in generating a new step.
            quantum_force(R_old, Q_force_old, i);

            // Move particle i slightly. Done with a loop so that each dimension gets its
            // own random noise.
            // Alt: R_new.col(i) = R_old.col(i) + (centered(rand_gen) * std::sqrt(_config.time_step)
            //                                     + 0.5 * _config.time_step * Q_force_old);
            for (int d = 0; d < _config.dims; ++d) {
                R_new(d, i) = R_old(d, i) + rnorm(rand_gen) * std::sqrt(_config.time_step)
                                          + 0.5 * _config.time_step * Q_force_old(d);
            }

            // Update distance matrix with the new postion.
            update_distance_matrix(i, R_new);

            // Calculate the new wavefunction.
            Real Psi_new = Psi(R_new);

            // Calculate the new quantum force.
            quantum_force(R_new, Q_force_new, i);

            // The ratio of the Greens funcitons: G(R_old, R_new, dt) / G(R_new, R_old, dt).
            double green = std::exp(0.25 * arma::dot(
                                        Q_force_old + Q_force_new,
                                        0.5 * _config.time_step * (Q_force_old-Q_force_new)
                                            + 2 * (R_old.col(i) - R_new.col(i))
                                        ));

            double acceptance_prob = green * (Psi_new * Psi_new) / (Psi_old * Psi_old);

            // New move accepted?
            if (unif(rand_gen) <= acceptance_prob) {
                // Update old <- new.
                Psi_old = Psi_new;
                Q_force_old = Q_force_new;
                R_old.col(i) = R_new.col(i);

                accepted_moves++;
            }
            else {
                // Restore new <- old.
                update_distance_matrix(i, R_old);
                R_new.col(i) = R_old.col(i);
            }

            // Update averages.
            Real E = E_local(R_new);
            E_sum += E;
            E2_sum += E*E;

            if (out != nullptr) {
                out->write(reinterpret_cast<const char*>(&E), sizeof(E));
            }

        }
    }

    // Compute final results.
    Real energy = E_sum / (n_cycles * _config.n_particles);
    Real energy_squared = E2_sum / (n_cycles * _config.n_particles);
    Real variance = energy_squared - energy*energy;
    Real acceptance_rate = accepted_moves / (Real) (n_cycles * _config.n_particles);
    return {energy, energy_squared, variance, _alpha, _beta, acceptance_rate};
}


} // namespace VMC
