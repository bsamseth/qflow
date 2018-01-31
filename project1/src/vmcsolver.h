#ifndef VMCSOLVER_H
#define VMCSOLVER_H

#include <random>
#include <cassert>
#include <armadillo>

namespace VMC {

std::default_random_engine rand_gen;
std::uniform_real_distribution<double> unif(0, 1);
std::uniform_real_distribution<double> centered(-0.5, 0.5);

namespace Constants {
} // namespace Constants

enum Dimensions {
    DIM_1 = 1, DIM_2, DIM_3
};
enum class HOType {
    SYMMETRIC, ELLIPTICAL
};
enum class InteractionType {
    OFF = 0, ON
};

enum class AnalyticAcceleration {
    OFF = 0, ON
};

template<Dimensions Dims, bool Squared>
double norm(const arma::mat &u) {
    double res = 0;
    for (int i = 0; i < Dims; ++i)
        res += u(i) * u(i);
    return Squared ? res : std::sqrt(res);
}

template<int N_Particles,
         Dimensions Dims,
         HOType T,
         InteractionType Interaction,
         AnalyticAcceleration Accelerate>
class VMCSolver {
private:
    const double _mass, _omega_ho, _omega_z, _a, _h, _h2, _step_length;
    double _alpha = 0.5, _beta = 1;

public:
    VMCSolver(double mass = 1, double omega_ho = 1, double omega_z = 1,
              double a = 0, double h = 0.001, double step_length = 1) :
        _mass(mass), _omega_ho(omega_ho*omega_ho),
        _omega_z(omega_z*omega_z), _a(a),
        _h(h), _h2(1/(h*h)), _step_length(step_length) {}

    double V_ext(const arma::mat &R) const {
        double pot = 0;
        for (int i = 0; i < R.n_rows; ++i) {
            if (T == HOType::ELLIPTICAL and Dims == Dimensions::DIM_3) {
                pot += _omega_ho * (R(i, 0)*R(i, 0) + R(i, 1)*R(i, 1))
                     + _omega_z  *  R(i, 2)*R(i, 2);
            } else {
                pot += _omega_ho * norm<Dims, true>(R.row(i));
            }
        }
        return 0.5 * _mass * pot;
    }

    double V_int(const arma::mat &R) const {
        if (Interaction == InteractionType::OFF) return 0;
        for (int i = 0; i < N_Particles; ++i) {
            for (int j = i + 1; j < N_Particles; ++j) {
                if (norm<Dims, true>(R.row(i) - R.row(j)) <= _a*_a)
                    return std::numeric_limits<double>::max();
            }
        }
        return 0;
    }

    double Psi_f(const arma::mat &R) const {
        if (Interaction == InteractionType::OFF) return 1;
        double f = 1;
        for (int i = 0; i < R.n_rows; ++i) {
            for (int j = i + 1; j < R.n_rows; ++j) {
                double r_ij = norm<Dims, false>(R.row(i) - R.row(j));
                if (r_ij <= _a)
                    return 0;
                f *= (1 - _a / r_ij);
            }
        }
        return f;
    }

    double Psi_g(const arma::mat &R) const {
        double g = 0;
        for (int i = 0; i < N_Particles; ++i) {
            g += norm<Dims, true>(R.row(i));
        }
        return std::exp(-_alpha * g);
    }

    double Psi(const arma::mat &R) const {
        return Psi_g(R) * Psi_f(R);
    }

    double E_kinetic(arma::mat &R) const {
        double Ek = 0;
        for (int i = 0; i < N_Particles; ++i) {
            for (int d = 0; d < Dims; ++d) {
                double old = Ek;
                Ek -= 2 * Psi(R);  // -2 * Psi(R)
                R(i, d) += _h;
                Ek += Psi(R);      // Psi(R + h)
                R(i, d) -= 2*_h;
                Ek += Psi(R);      // Psi(R - h)
                R(i, d) += _h;
                printf("i=%d: d^2/dx_%d^2 = %.10f\n", i, d, Ek - old);
            }
        }
        return -0.5 / _mass * Ek * _h2;
    }

    double E_local(arma::mat &R) const {
        if (Accelerate == AnalyticAcceleration::OFF)
            return E_kinetic(R) / Psi(R) + V_ext(R) + V_int(R);

        double E_L = 0;
        for (int k = 0; k < N_Particles; ++k) {
            const arma::mat::fixed<1, Dims> r_k = R.row(k);
            arma::mat::fixed<1, Dims> r_k_skewed = r_k;
            if (Dims == Dimensions::DIM_3)
                r_k_skewed(0, 2) *= _beta;
            E_L += 2*_alpha * (2*_alpha * norm<Dims, true>(r_k_skewed)
                               - (Dims == Dimensions::DIM_3 ? 2 + _beta : Dims));

            if (Interaction == InteractionType::OFF)
                continue;
            else
                assert(false);  // TODO
        }
        printf("E_L = %.10f + %.10f - 0.5 * %.10f = %.10f\n", V_ext(R), V_int(R), E_L,
               V_ext(R)+V_int(R)-0.5*E_L);
        return V_ext(R) + V_int(R) - 0.5 * E_L;
    }

    void run(const int n_cycles) const {
        arma::mat::fixed<N_Particles, Dims> R_old;
        arma::mat::fixed<N_Particles, Dims> R_new;
        double E_sum = 0, E2_sum = 0;
        // Random initial starting point.
        for (int i = 0; i < N_Particles; ++i) {
            for (int d = 0; d < Dims; ++d) {
                R_old(i, d) = R_new(i, d) = _step_length * centered(rand_gen);
            }
        }

        for (int cycle = 1; cycle <= n_cycles; ++cycle) {
            double Psi_old = Psi(R_old);
            for (int i = 0; i < N_Particles; ++i) {
                // Move particle i slightly.
                for (int d = 0; d < Dims; ++d) {
                    R_new(i, d) = R_old(i, d) + _step_length * centered(rand_gen);
                }
                double Psi_new = Psi(R_new);

                // New move accepted?
                if (unif(rand_gen) <= (Psi_new * Psi_new) / (Psi_old * Psi_old)) {
                    // Update old <- new.
                    Psi_old = Psi_new;
                    for (int d = 0; d < Dims; ++d) {
                        R_old(i, d) = R_new(i, d);
                    }
                }
                else {
                    // Restore new <- old.
                    for (int d = 0; d < Dims; ++d) {
                        R_new(i, d) = R_old(i, d);
                    }
                }

                // Update averages.
                double E = E_local(R_new);
                E_sum += E;
                E2_sum += E*E;

                printf("E = %.16f\n", E);
            }
        }
        double energy = E_sum / (n_cycles * N_Particles);
        double energy_squared = E2_sum / (n_cycles * N_Particles);
        double variance = energy_squared - energy*energy;
        std::cout << "Energy = " << energy;
        std::cout << " Energy (squared sum) = " << energy_squared;
        std::cout << " Variance = " << variance;
        std::cout << std::endl;
    }

};
} // namespace VMC
#endif // VMCSOLVER_H
