#include <iostream>
#include "definitions.hpp"
#include "system.hpp"
#include "wavefunction.hpp"
#include "simplegaussian.hpp"
#include "hamiltonian.hpp"
#include "sampler.hpp"
#include "optimizer.hpp"

#define SET_ALPHA(alpha) (wavefunction.set_params((alpha), wavefunction.get_beta(), wavefunction.get_alpha()))

namespace Optimizer {

Real gradient_decent_optimizer(Wavefunction &wavefunction,
                               const Hamiltonian &hamiltonian,
                               Sampler& sampler,
                               Real initial_guess,
                               Real learning_rate,
                               int sample_points,
                               int max_iterations,
                               Real dE_eps)
{
    Real alpha = initial_guess;

    int iteration = 0;

    Real E_L_der;

    do {
        SET_ALPHA(alpha);

        Real E_tot = 0;
        Real E_tot_sq = 0;
        Real psi_der_tot = 0;
        Real psi_der_E_tot = 0;

        for (int sample = 0; sample < sample_points; ++sample) {
            const System &system = sampler.next_configuration();
            Real E = hamiltonian.local_energy(system, wavefunction);
            Real psi_der = wavefunction.derivative_alpha(system);

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

        alpha = alpha - learning_rate * E_L_der;

        iteration++;

        printf("Iteration %d: alpha=%.10f, E=%.10f, var=%.10f, dE=%.10f\n", iteration, alpha, E_tot, variance, E_L_der);

    } while (iteration < max_iterations and std::abs(E_L_der) > dE_eps);// and std::abs(E - E_last) > E_eps and std::abs(dE - dE_last) > dE_eps);

    return wavefunction.get_alpha();
}

} // namespace Optimizer
