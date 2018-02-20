#include <random>
#include <cassert>
#include <omp.h>

#include "vmcsolver.hpp"

namespace VMC {

std::mt19937_64 rand_gen;
std::uniform_real_distribution<Real> unif(0, 1);
std::uniform_real_distribution<Real> centered(-0.5, 0.5);
std::normal_distribution<Real> rnorm(0, 1);

VMCSolver::VMCSolver(const VMCConfiguration &config) : _config(config) {
    R_old = arma::zeros<arma::Mat<Real>>(_config.dims, _config.n_particles);
    R_new = arma::zeros<arma::Mat<Real>>(_config.dims, _config.n_particles);
    dist  = arma::zeros<arma::Mat<Real>>(_config.n_particles, _config.n_particles);
}

void VMCSolver::initialize_distance_matrix(const arma::Mat<Real> &R) {

    // Initialize dist as an upper triangluar matrix of distances
    // based on the positions in R.

    for (int i = 0; i < _config.n_particles; ++i) {
        for (int j = i + 1; j < _config.n_particles; ++j) {
            dist(i, j) = arma::norm(R.col(i) - R.col(j));
        }
    }
}

void VMCSolver::update_distance_matrix(int particle, const arma::Mat<Real> &R) {

    // Update row particle. Only work on upper half of matrix.
    for (int other = particle + 1; other < _config.n_particles; ++other) {
        dist(particle, other) = arma::norm(R.col(particle) - R.col(other));
    }

    // Update col particle. Only work on upper half of matrix.
    for (int other = 0; other < particle; ++other) {
        dist(other, particle) = arma::norm(R.col(other) - R.col(particle));
    }
}

Real VMCSolver::V_ext(const arma::Mat<Real> &R) const {
    Real pot = 0;
    for (int i = 0; i < _config.n_particles; ++i) {
        if (_config.ho_type == HOType::ELLIPTICAL and _config.dims == Dimensions::DIM_3) {
            pot += _config.omega_ho * (R(0, i)*R(0, i) + R(1, i)*R(1, i))
                 + _config.omega_z  *  R(2, i)*R(2, i);
        } else {
            pot += _config.omega_ho * arma::dot(R.col(i), R.col(i));
        }
    }
    return 0.5 * pot;
}

Real VMCSolver::V_int() const {
    if (_config.interaction == InteractionType::OFF) return 0;
    for (int i = 0; i < _config.n_particles; ++i) {
        for (int j = i + 1; j < _config.n_particles; ++j) {
            if (dist(i, j) <= _config.a)
                return std::numeric_limits<Real>::max();
        }
    }
    return 0;
}

Real VMCSolver::Psi_f() const {
    if (_config.interaction == InteractionType::OFF) return 1;
    Real f = 1;
    for (int i = 0; i < _config.n_particles; ++i) {
        for (int j = i + 1; j < _config.n_particles; ++j) {
            const Real r_ij = dist(i, j);
            if (r_ij <= _config.a)
                return 0;
            f *= (1 - _config.a / r_ij);
        }
    }
    return f;
}

Real VMCSolver::Psi_g(const arma::Mat<Real> &R) const {
    Real g = 0;
    for (int i = 0; i < _config.n_particles; ++i) {
        g += arma::dot(R.col(i), R.col(i));
    }
    return std::exp(-_alpha * g);
}

Real VMCSolver::Psi(const arma::Mat<Real> &R) const {
    return Psi_g(R) * Psi_f();
}

Real VMCSolver::E_kinetic(arma::Mat<Real> &R) {

    Real Ek = - 2 * (_config.n_particles * _config.dims) * Psi(R);

    for (int i = 0; i < _config.n_particles; ++i) {
        for (int d = 0; d < _config.dims; ++d) {

            // Storing a temporary instead of adding/subtracting away
            // the changes should help avoid accumulating rounding errors.
            const auto temp = R(d, i);

            // Psi(R) depends on the dist matrix being updated after changes.
            // Only needed if interaction is on.

            R(d, i) = temp + _config.h;
            if (_config.interaction == InteractionType::ON) update_distance_matrix(i, R);

            Ek += Psi(R);      // Psi(R + h)

            R(d, i) = temp - _config.h;
            if (_config.interaction == InteractionType::ON) update_distance_matrix(i, R);

            Ek += Psi(R);      // Psi(R - h)

            // Reset distance.
            R(d, i) = temp;
            if (_config.interaction == InteractionType::ON) update_distance_matrix(i, R);
        }
    }
    return -0.5 * Ek * _config.h2;
}

Real VMCSolver::E_local(arma::Mat<Real> &R) {
    if (_config.acceleration == AnalyticAcceleration::OFF)
        return E_kinetic(R) / Psi(R) + V_ext(R) + V_int();

    Real E_L = 0;
    const bool no_interaction = _config.interaction == InteractionType::OFF;
    const int one_body_beta_term = - (_config.dims == Dimensions::DIM_3 ?
                                          2 + _beta : (int) _config.dims);

    for (int k = 0; k < _config.n_particles; ++k) {

        // r_k = R(:, k)
        // r_k_skewed = {x_k, y_k, _beta * z_k} if Dims == 3
        //               r_k                    otherwise.
        const auto r_k = R.col(k);
        arma::Col<Real> r_k_skewed = R.col(k);
        if (_config.dims == Dimensions::DIM_3)
            r_k_skewed(2) *= _beta;


        // First term, no interaction.
        E_L += 2*_alpha * (2*_alpha * arma::dot(r_k_skewed, r_k_skewed) + one_body_beta_term);


        // Remaining terms are only for interaction.
        if (no_interaction)
            continue;


        arma::Col<Real> term (_config.dims);
        for (int j = 0; j < _config.n_particles; ++j) {  // j != k.
            if (j == k) continue;

            const auto r_kj = r_k - R.col(j);
            const Real r_kj_norm = dist(std::min(k, j), std::max(k, j));
            const Real r_kj_2 = r_kj_norm * r_kj_norm;

            term += r_kj * (_config.a / (r_kj_2 * (r_kj_norm - _config.a)));

            E_L += _config.a * (_config.a - 2 * r_kj_norm) / (r_kj_2 * (r_kj_norm - _config.a) * (r_kj_norm - _config.a))
                    + 2 * _config.a / (r_kj_2 * (r_kj_norm - _config.a));

            for (int i = 0; i < _config.n_particles; ++i) {  // i != k.
                if (i == k) continue;

                const auto r_ki = r_k - R.col(i);
                const Real r_ki_norm = dist(std::min(k, i), std::max(k, i));
                const Real r_ki_2 = r_ki_norm * r_ki_norm;

                E_L += arma::dot(r_ki, r_kj) * (_config.a * _config.a / (r_ki_2 * r_kj_2 * (r_ki_norm - _config.a) * (r_kj_norm - _config.a)));
            }
        }
        E_L -= 4 * _alpha * arma::dot(r_k_skewed, term);
    }
    return V_ext(R) + V_int() - 0.5 * E_L;
}

Results VMCSolver::run_MC(const int n_cycles) {
    arma::Mat<Real> R_old (_config.dims, _config.n_particles);
    arma::Mat<Real> R_new (_config.dims, _config.n_particles);
    Real E_sum = 0, E2_sum = 0;

    // Random initial starting point.
    for (int i = 0; i < _config.n_particles; ++i) {
        for (int d = 0; d < _config.dims; ++d) {
            R_old(d, i) = R_new(d, i) = _config.step_length * centered(rand_gen);
        }
    }

    initialize_distance_matrix(R_old);

    int accepted_moves = 0;
    for (int cycle = 1; cycle <= n_cycles; ++cycle) {
        Real Psi_old = Psi(R_old);
        for (int i = 0; i < _config.n_particles; ++i) {
            // Move particle i slightly.
            for (int d = 0; d < _config.dims; ++d) {
                R_new(d, i) = R_old(d, i) + _config.step_length * centered(rand_gen);
            }

            update_distance_matrix(i, R_new);

            Real Psi_new = Psi(R_new);

            // New move accepted?
            if (unif(rand_gen) <= (Psi_new * Psi_new) / (Psi_old * Psi_old)) {
                accepted_moves++;
                // Update old <- new.
                Psi_old = Psi_new;
                for (int d = 0; d < _config.dims; ++d) {
                    R_old(d, i) = R_new(d, i);
                }
            }
            else {
                // Restore new <- old.
                update_distance_matrix(i, R_old);
                for (int d = 0; d < _config.dims; ++d) {
                    R_new(d, i) = R_old(d, i);
                }
            }

            // Update averages.
            Real E = E_local(R_new);
            E_sum += E;
            E2_sum += E*E;
        }
    }
    Real energy = E_sum / (n_cycles * _config.n_particles);
    Real energy_squared = E2_sum / (n_cycles * _config.n_particles);
    Real variance = energy_squared - energy*energy;
    Real acceptance_rate = accepted_moves / (Real) (n_cycles * _config.n_particles);
    return {energy, energy_squared, variance, _alpha, _beta, acceptance_rate};
}

Results VMCSolver::vmc(const int n_cycles,
            std::ostream &out,
            const Real alpha_min,
            const Real alpha_max,
            const int alpha_n,
            const Real beta_min,
            const Real beta_max,
            const int beta_n) {

    // Used to store best results.
    Results best;
    best.variance = std::numeric_limits<Real>::max();

    // Define variational space.;
    const auto alphas = arma::linspace<arma::Col<Real>>(alpha_min, alpha_max, alpha_n);
    const auto betas  = arma::linspace<arma::Col<Real>>(beta_min, beta_max, beta_n);

    // Write header to stream.
    out << "# alpha beta <E> <E^2>\n";

    // For every combination of parameters,
    // write the results of MC to the stream.
    for (const auto &alpha : alphas) {
        _alpha = alpha;
        for (const auto &beta : betas) {
            _beta = beta;
            Results res = run_MC(n_cycles);
            out << _alpha << " "
                << _beta  << " "
                << res.E  << " "
                << res.E2 << "\n";

            // New best parameter choice?
            if (res.variance < best.variance)
                best = res;
        }
    }
    // Flush when you are done.
    out << std::flush;

    return best;
}

}  // namespace VMC
