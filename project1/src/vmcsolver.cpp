#include <random>
#include <cassert>

#include "vmcsolver.hpp"

namespace VMC {

std::mt19937_64 rand_gen;
std::uniform_real_distribution<double> unif(0, 1);
std::uniform_real_distribution<double> centered(-0.5, 0.5);
std::normal_distribution<double> rnorm(0, 1);

VMCSolver::VMCSolver(const VMCConfiguration &config) : _config(config) {
    R_old = arma::zeros<arma::mat>(_config.n_particles, _config.dims);
    R_new = arma::zeros<arma::mat>(_config.n_particles, _config.dims);
    dist  = arma::zeros<arma::mat>(_config.n_particles, _config.n_particles);
}

void VMCSolver::initialize_distance_matrix(const arma::mat &R) {

    // Initialize dist as an upper triangluar matrix of distances
    // based on the positions in R.

    for (int i = 0; i < _config.n_particles; ++i) {
        for (int j = i + 1; j < _config.n_particles; ++j) {
            dist(i, j) = arma::norm(R.row(i) - R.row(j));
        }
    }
}

void VMCSolver::update_distance_matrix(int particle, const arma::mat &R) {

    // Update row particle. Only work on upper half of matrix.
    for (int other = particle + 1; other < _config.n_particles; ++other) {
        dist(particle, other) = arma::norm(R.row(particle) - R.row(other));
    }

    // Update col particle. Only work on upper half of matrix.
    for (int other = 0; other < particle; ++other) {
        dist(other, particle) = arma::norm(R.row(other) - R.row(particle));
    }
}

double VMCSolver::V_ext(const arma::mat &R) const {
    double pot = 0;
    for (unsigned i = 0; i < R.n_rows; ++i) {
        if (_config.ho_type == HOType::ELLIPTICAL and _config.dims == Dimensions::DIM_3) {
            pot += _config.omega_ho * (R(i, 0)*R(i, 0) + R(i, 1)*R(i, 1))
                 + _config.omega_z  *  R(i, 2)*R(i, 2);
        } else {
            pot += _config.omega_ho * arma::dot(R.row(i), R.row(i));
        }
    }
    return 0.5 * pot;
}

double VMCSolver::V_int(const arma::mat &R) const {
    if (_config.interaction == InteractionType::OFF) return 0;
    for (unsigned i = 0; i < R.n_rows; ++i) {
        for (unsigned j = i + 1; j < R.n_rows; ++j) {
            if (dist(i, j) <= _config.a)
                return std::numeric_limits<double>::max();
        }
    }
    return 0;
}

double VMCSolver::Psi_f(const arma::mat &R) const {
    if (_config.interaction == InteractionType::OFF) return 1;
    double f = 1;
    for (unsigned i = 0; i < R.n_rows; ++i) {
        for (unsigned j = i + 1; j < R.n_rows; ++j) {
            const double r_ij = dist(i, j);
            if (r_ij <= _config.a)
                return 0;
            f *= (1 - _config.a / r_ij);
        }
    }
    return f;
}

double VMCSolver::Psi_g(const arma::mat &R) const {
    double g = 0;
    for (unsigned i = 0; i < R.n_rows; ++i) {
        g += arma::dot(R.row(i), R.row(i));
    }
    return std::exp(-_alpha * g);
}

double VMCSolver::Psi(const arma::mat &R) const {
    return Psi_g(R) * Psi_f(R);
}

double VMCSolver::E_kinetic(arma::mat &R) {

    double Ek = - 2 * (_config.n_particles * _config.dims) * Psi(R);

    for (unsigned i = 0; i < R.n_rows; ++i) {
        for (unsigned d = 0; d < R.n_cols; ++d) {

            // Storing a temporary instead of adding/subtracting away
            // the changes should help avoid accumulating rounding errors.
            const auto temp = R(i, d);

            // Psi(R) depends on the dist matrix being updated after changes.
            // Only needed if interaction is on.

            R(i, d) = temp + _config.h;
            if (_config.interaction == InteractionType::ON) update_distance_matrix(i, R);

            Ek += Psi(R);      // Psi(R + h)

            R(i, d) = temp - _config.h;
            if (_config.interaction == InteractionType::ON) update_distance_matrix(i, R);

            Ek += Psi(R);      // Psi(R - h)

            // Reset distance.
            R(i, d) = temp;
            if (_config.interaction == InteractionType::ON) update_distance_matrix(i, R);
        }
    }
    return -0.5 * Ek * _config.h2;
}

double VMCSolver::E_local(arma::mat &R) {
    if (_config.acceleration == AnalyticAcceleration::OFF)
        return E_kinetic(R) / Psi(R) + V_ext(R) + V_int(R);

    double E_L = 0;
    for (unsigned k = 0; k < R.n_rows; ++k) {

        // r_k = R(k, :)
        // r_k_skewed = {x_k, y_k, _beta * z_k} if Dims == 3
        //               r_k                    otherwise.
        const auto r_k = R.row(k);
        arma::rowvec r_k_skewed = R.row(k);
        if (_config.dims == Dimensions::DIM_3)
            r_k_skewed(2) *= _beta;


        // First term, no interaction.
        E_L += 2*_alpha * (2*_alpha * arma::dot(r_k_skewed, r_k_skewed)
                           - (_config.dims == Dimensions::DIM_3 ? 2 + _beta :
                                                                  _config.dims));


        // Remaining terms are only for interaction.
        if (_config.interaction == InteractionType::OFF)
            continue;


        arma::rowvec term (R.n_cols);
        for (unsigned j = 0; j < R.n_rows; ++j) {  // j != k.
            if (j == k) continue;

            const auto r_kj = r_k - R.row(j);
            const double r_kj_norm = dist(std::min(k, j), std::max(k, j));
            const double r_kj_2 = r_kj_norm * r_kj_norm;

            term += r_kj * (_config.a / (r_kj_2 * (r_kj_norm - _config.a)));

            E_L += _config.a * (_config.a - 2 * r_kj_norm) / (r_kj_2 * (r_kj_norm - _config.a) * (r_kj_norm - _config.a))
                    + 2 * _config.a / (r_kj_2 * (r_kj_norm - _config.a));

            for (unsigned i = 0; i < R.n_rows; ++i) {  // i != k.
                if (i == k) continue;

                const auto r_ki = r_k - R.row(i);
                const double r_ki_norm = dist(std::min(k, i), std::max(k, i));
                const double r_ki_2 = r_ki_norm * r_ki_norm;

                E_L += arma::dot(r_ki, r_kj) * (_config.a * _config.a / (r_ki_2 * r_kj_2 * (r_ki_norm - _config.a) * (r_kj_norm - _config.a)));
            }
        }
        E_L -= 4 * _alpha * arma::dot(r_k_skewed, term);
    }
    return V_ext(R) + V_int(R) - 0.5 * E_L;
}

Results VMCSolver::run_MC(const int n_cycles) {
    arma::mat R_old (_config.n_particles, _config.dims);
    arma::mat R_new (arma::size(R_old));
    double E_sum = 0, E2_sum = 0;

    // Random initial starting point.
    for (int i = 0; i < _config.n_particles; ++i) {
        for (int d = 0; d < _config.dims; ++d) {
            R_old(i, d) = R_new(i, d) = _config.step_length * centered(rand_gen);
        }
    }

    initialize_distance_matrix(R_old);

    int accepted_moves = 0;
    for (int cycle = 1; cycle <= n_cycles; ++cycle) {
        double Psi_old = Psi(R_old);
        for (int i = 0; i < _config.n_particles; ++i) {
            // Move particle i slightly.
            for (int d = 0; d < _config.dims; ++d) {
                R_new(i, d) = R_old(i, d) + _config.step_length * centered(rand_gen);
            }

            update_distance_matrix(i, R_new);

            double Psi_new = Psi(R_new);

            // New move accepted?
            if (unif(rand_gen) <= (Psi_new * Psi_new) / (Psi_old * Psi_old)) {
                accepted_moves++;
                // Update old <- new.
                Psi_old = Psi_new;
                for (int d = 0; d < _config.dims; ++d) {
                    R_old(i, d) = R_new(i, d);
                }
            }
            else {
                // Restore new <- old.
                update_distance_matrix(i, R_old);
                for (int d = 0; d < _config.dims; ++d) {
                    R_new(i, d) = R_old(i, d);
                }
            }

            // Update averages.
            double E = E_local(R_new);
            E_sum += E;
            E2_sum += E*E;
        }
    }
    double energy = E_sum / (n_cycles * _config.n_particles);
    double energy_squared = E2_sum / (n_cycles * _config.n_particles);
    double variance = energy_squared - energy*energy;
    double acceptance_rate = accepted_moves / (double) (n_cycles * _config.n_particles);
    return {energy, energy_squared, variance, _alpha, _beta, acceptance_rate};
}

Results VMCSolver::vmc(
            const int n_cycles,
            std::ostream &out,
            const double alpha_min,
            const double alpha_max,
            const double alpha_n,
            const double beta_min,
            const double beta_max,
            const double beta_n) {

    // Used to store best results.
    Results best;
    best.variance = std::numeric_limits<double>::max();

    // Define variational space.;
    const auto alphas = arma::linspace<arma::vec>(alpha_min, alpha_max, alpha_n);
    const auto betas  = arma::linspace<arma::vec>(beta_min, beta_max, beta_n);

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
