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
    // Thermalize the sampler.
    for (int run = 0; run < sample_points; ++run) {
        sampler.next_configuration();
    }


    Real alpha = initial_guess;

    int iteration = 0;

    Real E_L_der, E_L_der_prev = 1;

    do {
        wavefunction.set_params(alpha, wavefunction.get_beta(), wavefunction.get_a());

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

    return wavefunction.get_alpha();
}

} // namespace Optimizer
