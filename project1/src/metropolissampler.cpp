#include "definitions.hpp"
#include "system.hpp"
#include "wavefunction.hpp"
#include "hamiltonian.hpp"
#include "metropolissampler.hpp"

MetropolisSampler::MetropolisSampler(const System &system,
                                     const Wavefunction &wavefunction,
                                     Real step)
    : Sampler(system, wavefunction, step)
{
    initialize_system();
}

void MetropolisSampler::initialize_system() {
    for (int i = 0; i < _system_old.get_n_bosons(); ++i) {
        for (int d = 0; d < _system_old.get_dimensions(); ++d) {
            _system_old[i][d] = _step * centered(rand_gen);
        }
    }
    _system_new = _system_old;
}

void MetropolisSampler::perturb_system() {
    Boson &boson = _system_new[_particle_to_move];

    for (int d = 0; d < boson.get_dimensions(); ++d) {
        boson[d] += _step * centered(rand_gen);
    }

    _psi_new = (*_wavefunction)(_system_new);
}

Real MetropolisSampler::acceptance_probability() {
    return square(_psi_new) / square(_psi_old);
}
