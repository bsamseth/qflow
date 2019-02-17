#include "definitions.hpp"
#include "system.hpp"
#include "wavefunction.hpp"
#include "hamiltonian.hpp"
#include "importancesampler.hpp"

ImportanceSampler::ImportanceSampler(const System &system,
                                     Wavefunction &wavefunction,
                                     Real step)
    : Sampler(system, wavefunction, step)
{
    for (std::size_t i = 0; i < _N_instances; ++i) {
        initialize_system(i);
    }
}

void ImportanceSampler::initialize_system(std::size_t i) {
    StateInfo& state = _instances[i];
    for (int i = 0; i < state.system_old.rows(); ++i) {
        for (int d = 0; d < state.system_old.cols(); ++d) {
            state.system_old(i, d) = rnorm(rand_gen) * std::sqrt(_step);
        }
    }
    state.system_new = state.system_old;
    state.psi_old = (*_wavefunction)(state.system_old);
}

void ImportanceSampler::perturb_system(std::size_t i) {
    StateInfo& state = _instances[i];
    for (int d = 0; d < state.system_new.cols(); ++d) {
        state.system_new(state.particle_to_move, d) += rnorm(rand_gen) * std::sqrt(_step)
                   + 0.5 * _step * _wavefunction->drift_force(state.system_new, state.particle_to_move, d);
    }

    state.psi_new = (*_wavefunction)(state.system_new);
}

Real ImportanceSampler::acceptance_probability(std::size_t i) const {
    const StateInfo& state = _instances[i];
    const RowVector &r_old = state.system_old.row(state.particle_to_move);
    const RowVector &r_new = state.system_new.row(state.particle_to_move);

    Real green1 = 0, green2 = 0;
    for (int d = 0; d < r_old.size(); ++d) {
        green1 += square(r_old[d] - r_new[d] - 0.5 * _step * _wavefunction->drift_force(state.system_new, state.particle_to_move, d));
        green2 += square(r_new[d] - r_old[d] - 0.5 * _step * _wavefunction->drift_force(state.system_old, state.particle_to_move, d));
    }

    // Ratio = exp(-green1/(4*D*step)) / exp(-green2/(4*D*step))
    Real ratio = std::exp( (green2 - green1) / (2 * _step) );

    return ratio * square(state.psi_new) / square(state.psi_old);
}
