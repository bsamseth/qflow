#ifndef VMCSOLVER_HPP
#define VMCSOLVER_HPP

#include <ostream>
#include <iomanip>
#include <armadillo>

namespace VMC {

using Real = double;

extern std::mt19937_64 rand_gen;
extern std::uniform_real_distribution<Real> unif;
extern std::uniform_real_distribution<Real> centered;
extern std::normal_distribution<Real> rnorm;


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

struct Results {
    Real E, E2, variance, alpha, beta, acceptance_rate;
};

struct VMCConfiguration {
    int n_particles;
    Dimensions dims;
    HOType ho_type;
    InteractionType interaction;
    AnalyticAcceleration acceleration;
    Real omega_ho;
    Real omega_z;
    Real a;
    Real h;
    Real h2;
    Real step_length;
    Real time_step;
};

class VMCSolver {
protected:
    const VMCConfiguration _config;
    Real _alpha = 0.5, _beta = 1;
    arma::Mat<Real> dist, R_old, R_new;

public:
    VMCSolver(const VMCConfiguration &config);

    void initialize_distance_matrix(const arma::Mat<Real> &R);

    void update_distance_matrix(int particle, const arma::Mat<Real> &R);

    Real V_ext(const arma::Mat<Real> &R) const;

    Real V_int() const;

    Real Psi_f() const;

    Real Psi_g(const arma::Mat<Real> &R) const;

    Real Psi(const arma::Mat<Real> &R) const;

    Real E_kinetic(arma::Mat<Real> &R);

    Real E_local(arma::Mat<Real> &R);

    virtual Results run_MC(const int n_cycles);

    Results vmc(const int n_cycles,
                std::ostream &out,
                const Real alpha_min,
                const Real alpha_max,
                const int alpha_n,
                const Real beta_min = 1,
                const Real beta_max = 1,
                const int beta_n = 1);
};


}  // namespace VMC

inline std::ostream& operator<<(std::ostream &strm, const VMC::Results &r) {
    strm.precision(6);
    strm << std::scientific;
    return strm << "E = "  << r.E  << ", "
                << "E2 = " << r.E2 << ", "
                << "Var = " << r.variance << ", "
                << "alpha = " << r.alpha << ", "
                << "beta = " << r.beta << ", "
                << "acceptance rate = "
                << std::fixed << std::setprecision(3) <<  r.acceptance_rate;
}



#endif // VMCSOLVER_HPP
