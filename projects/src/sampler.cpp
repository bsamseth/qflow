#include <cassert>
#include <cmath>

#include "definitions.hpp"
#include "system.hpp"
#include "wavefunction.hpp"
#include "hamiltonian.hpp"
#include "sampler.hpp"

Sampler::Sampler(const System &system,
                 const Wavefunction &wavefunction,
                 Real step)
                : _step(step),
                  _wavefunction(&wavefunction),
                  _system_old(system),
                  _system_new(system)
{
    _psi_new = _psi_old = (*_wavefunction)(_system_old);

    // Some badly initialized systems/wavefunctions may give
    // NaNs out, which can screw up all subsequent calculations.
    // Catch this now and emulate a very unlikely state.
    if (std::isnan(_psi_old )) {
        _psi_new = _psi_old = 1e-15;
    }
}

System &Sampler::next_configuration() {

    perturb_system();

    if (unif(rand_gen) <= acceptance_probability()) {
        _accepted_steps++;
        _system_old.col(_particle_to_move) = _system_new.col(_particle_to_move);
        _psi_old = _psi_new;
    } else {
        _system_new.col(_particle_to_move) = _system_old.col(_particle_to_move);
    }

    assert(_system_old == _system_new);

    prepare_for_next_run();

    return _system_new;
}
