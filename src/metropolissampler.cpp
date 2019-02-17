#include "definitions.hpp"
#include "system.hpp"
#include "wavefunction.hpp"
#include "hamiltonian.hpp"
#include "metropolissampler.hpp"

MetropolisSampler::MetropolisSampler(const System &system,
                                     Wavefunction &wavefunction,
                                     Real step,
                                     std::size_t N)
    : Sampler(system, wavefunction, step, N)
{
    for (std::size_t i = 0; i < _N_instances; ++i) {
        initialize_system(i);
    }
}

void MetropolisSampler::initialize_system(std::size_t i) {
    StateInfo& state = _instances[i];
    for (int i = 0; i < state.system_old.rows(); ++i) {
        for (int d = 0; d < state.system_old.cols(); ++d) {
            state.system_old(i, d) = _step * centered(rand_gen);
        }
    }
    state.system_new = state.system_old;
    state.psi_old = (*_wavefunction)(state.system_old);
}

void MetropolisSampler::perturb_system(std::size_t i) {
    StateInfo& state = _instances[i];
    for (int d = 0; d < state.system_new.cols(); ++d) {
        state.system_new(state.particle_to_move, d) += _step * centered(rand_gen);
    }
    state.psi_new = (*_wavefunction)(state.system_new);
}

Real MetropolisSampler::acceptance_probability(std::size_t i) const {
    return square(_instances[i].psi_new) / square(_instances[i].psi_old);
}
