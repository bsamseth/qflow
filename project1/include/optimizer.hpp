#pragma once

#include "definitions.hpp"
#include "system.hpp"
#include "wavefunction.hpp"
#include "hamiltonian.hpp"
#include "sampler.hpp"

namespace Optimizer {

Real gradient_decent_optimizer(Wavefunction&,
                               const Hamiltonian&,
                               Sampler&,
                               Real initial_guess,
                               Real learning_rate = 0.1,
                               int sample_points = 10000,
                               int max_iterations = 100,
                               Real dE_eps = 1e-8);
}
