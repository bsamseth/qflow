#include "metropolissampler.hpp"

#include "definitions.hpp"
#include "hamiltonian.hpp"
#include "system.hpp"
#include "wavefunction.hpp"

MetropolisSampler::MetropolisSampler(const System& system,
                                     Wavefunction& wavefunction,
                                     Real          step)
    : Sampler(system, wavefunction, step)
{
    initialize_system();
}

void MetropolisSampler::initialize_system()
{
    for (int i = 0; i < _system_old.rows(); ++i)
    {
        for (int d = 0; d < _system_old.cols(); ++d)
        {
            _system_old(i, d) = _step * centered(rand_gen);
        }
    }
    _system_new = _system_old;
    _psi_old    = (*_wavefunction)(_system_old);
}

void MetropolisSampler::perturb_system()
{
    for (int d = 0; d < _system_new.cols(); ++d)
    {
        _system_new(_particle_to_move, d) += _step * centered(rand_gen);
    }
    _psi_new = (*_wavefunction)(_system_new);
}

Real MetropolisSampler::acceptance_probability() const
{
    return square(_psi_new) / square(_psi_old);
}
