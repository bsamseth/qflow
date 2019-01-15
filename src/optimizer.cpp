#include <iostream>
#include "definitions.hpp"
#include "system.hpp"
#include "wavefunction.hpp"
#include "simplegaussian.hpp"
#include "hamiltonian.hpp"
#include "sampler.hpp"
#include "optimizer.hpp"


namespace Optimizer {

Real gradient_decent_optimizer(Wavefunction &wavefunction,
        const Hamiltonian &hamiltonian,
        Sampler& sampler,
        Real initial_guess,
        Real learning_rate,
        int sample_points,
        int max_iterations,
        Real dE_eps,
        bool verbose)
{
    Real alpha = initial_guess;

    int iteration = 0;

    Real E_L_der, E_L_der_prev = 1;

    do {
        wavefunction.get_parameters()[0] = alpha;

        // Thermalize the sampler to the alpha guess.
        for (int run = 0; run < sample_points; ++run) {
            sampler.next_configuration();
        }

        Real E_tot = 0;
        Real E_tot_sq = 0;
        Real psi_der_tot = 0;
        Real psi_der_E_tot = 0;

        for (int sample = 0; sample < sample_points; ++sample) {
            System &system = sampler.next_configuration();
            Real E = hamiltonian.local_energy(system, wavefunction);
            Real psi_der = wavefunction.gradient(system)[0];

            E_tot += E;
            E_tot_sq += square(E);
            psi_der_tot += psi_der;
            psi_der_E_tot += psi_der * E;
        }

        E_tot /= sample_points;
        E_tot_sq /= sample_points;
        psi_der_tot /= sample_points;
        psi_der_E_tot /= sample_points;

        Real variance = E_tot_sq - square(E_tot);
        E_L_der = 2 * (psi_der_E_tot - psi_der_tot * E_tot);

        // If derivative changes sign, this indicates that we jumped over the minimum.
        // In this case, let's make smaller steps.
        if (E_L_der * E_L_der_prev < 0) {
            learning_rate /= 2;
        }

        E_L_der_prev = E_L_der;

        alpha = alpha - learning_rate * E_L_der;

        iteration++;

        if (verbose)
            printf("Iteration %d: alpha=%.10f, E=%.10f, var=%.10f, dE=%.10f, lr=%g\n", iteration, alpha, E_tot, variance, E_L_der, learning_rate);

    } while (iteration < max_iterations and std::abs(E_L_der) > dE_eps);

    return wavefunction.get_parameters()[0];
}

} // namespace Optimizer


SgdOptimizer::SgdOptimizer(Real eta) : _eta(eta) {}

RowVector SgdOptimizer::update_term(const RowVector &gradient) {
    return - _eta * gradient;
}

AdamOptimizer::AdamOptimizer(std::size_t n_parameters, Real alpha, Real beta1, Real beta2, Real epsilon)
    : _alpha(alpha),
    _beta1(beta1),
    _beta2(beta2),
    _epsilon(epsilon),
    _t(0),
    _m(RowVector::Zero(n_parameters)),
    _v(RowVector::Zero(n_parameters))
{ }

RowVector AdamOptimizer::update_term(const RowVector &gradient) {
    ++_t;
    _m = _beta1 * _m + (1 - _beta1) * gradient;
    _v = _beta2 * _v + (1 - _beta2) * (gradient.cwiseProduct(gradient));
    _alpha_t = _alpha * std::sqrt(1 - std::pow(_beta2, _t)) / (1 - std::pow(_beta1, _t));

    RowVector update(_m.size());
    for (int i = 0; i < _m.size(); ++i) {
        update[i] = -_alpha_t * _m[i] / (std::sqrt(_v[i]) + _epsilon);
    }
    return update;
}






