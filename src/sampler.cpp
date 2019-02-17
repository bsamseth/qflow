#include <cassert>
#include <cmath>

#include "definitions.hpp"
#include "system.hpp"
#include "wavefunction.hpp"
#include "hamiltonian.hpp"
#include "sampler.hpp"
#include "mpiutil.hpp"

Sampler::StateInfo::StateInfo(System s, Wavefunction& psi) : system_old(s), system_new(s)
{
    psi_old = psi_new = psi(s);

    // Some badly initialized systems/wavefunctions may give
    // NaNs out, which can screw up all subsequent calculations.
    // Catch this now and emulate a very unlikely state.
    if (std::isnan(psi_old)) {
        psi_new = psi_old = 1e-15;
    }
}

Sampler::Sampler(const System &system,
                 Wavefunction &wavefunction,
                 Real step)
                : _step(step),
                  _wavefunction(&wavefunction),
                  _N_instances(mpiutil::proc_count())

{
    for (std::size_t i = 0; i < _N_instances; ++i) {
        _instances.push_back({system, wavefunction});
    }

}

System &Sampler::next_configuration(std::size_t i) {

    perturb_system(i);

    StateInfo& state = _instances[i];
    if (unif(rand_gen) <= acceptance_probability(i)) {
        state.accepted_steps++;
        state.system_old.row(state.particle_to_move) = state.system_new.row(state.particle_to_move);
        state.psi_old = state.psi_new;
    } else {
        state.system_new.row(state.particle_to_move) = state.system_old.row(state.particle_to_move);
    }

    assert(state.system_old == state.system_new);

    prepare_for_next_run(i);

    return state.system_new;
}
