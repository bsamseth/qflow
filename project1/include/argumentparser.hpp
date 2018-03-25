#pragma once

#include <cstdlib>

#include "wavefunction.hpp"
#include "hamiltonian.hpp"
#include "sampler.hpp"
#include "calculator.hpp"


void parse_arguments(int argc, char **argv, Wavefunction **psi, Hamiltonian **H, Sampler **sampler, Calculator **calc, int *cycles);
void parse_arguments_optimize(int argc, char **argv, Wavefunction **psi, Hamiltonian **H, Sampler **sampler, Real *alpha_guess, Real *learning_rate, int *n_cycles, Real *minimum_gradient);
