#pragma once

#include "definitions.hpp"
#include "system.hpp"
#include "wavefunction.hpp"
#include "hamiltonian.hpp"
#include "sampler.hpp"

namespace Optimizer {

/**
 * Return an optimized value for alpha.
 * @param wavefunction Wavefunction to calculate for.
 * @param hamiltonian Hamiltonian to calculate for.
 * @param sampler Source of sampled System instances.
 * @param initial_guess Starting point for alpha.
 * @param learning_rate Learning rate to use.
 * @param sample_points Number of samples to use per iteration.
 * @param max_iterations Maximum number of iterations to run.
 * @param dE_eps Gradient value for which to accept as zero.
 * @param verbose True if interim output should be printed.
 * @return Optimized value for alpha.
 */
Real gradient_decent_optimizer(Wavefunction &wavefunction,
                               const Hamiltonian &hamiltonian,
                               Sampler &sampler,
                               Real initial_guess,
                               Real learning_rate = 0.1,
                               int sample_points = 10000,
                               int max_iterations = 100,
                               Real dE_eps = 1e-8,
                               bool verbose = true);
}
