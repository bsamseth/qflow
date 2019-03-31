#include "gibbssampler.hpp"

#include "definitions.hpp"
#include "distance.hpp"
#include "hamiltonian.hpp"
#include "system.hpp"
#include "wavefunction.hpp"

#include <cmath>

GibbsSampler::GibbsSampler(const System& system, RBMWavefunction& wavefunction)
    : Sampler(system, wavefunction, 0), _rbm(wavefunction), _system(system)
{
    initialize_system();
}

void GibbsSampler::initialize_system()
{
    _stddev = std::sqrt(_rbm._sigma2);
    std::normal_distribution<Real> dist(0, _stddev);
    for (int i = 0; i < _system.rows(); ++i)
    {
        for (int d = 0; d < _system.cols(); ++d)
        {
            _system(i, d) = dist(rand_gen);
        }
    }
}

System& GibbsSampler::next_configuration()
{
    RowVector h(_rbm._N);
    for (int j = 0; j < _rbm._N; ++j)
    {
        h[j] = unif(rand_gen) < P_h(j, _system) ? 1 : 0;
    }

    for (int i = 0; i < _rbm._M; ++i)
    {
        Real mean = _rbm.get_parameters()[_rbm.a(i)];
        for (int j = 0; j < _rbm._N; ++j)
        {
            mean += _rbm.get_parameters()[_rbm.w(i, j)] * h[j];
        }

        std::normal_distribution<Real> dist(mean, _stddev);
        _system.data()[i] = dist(rand_gen);
    }

    for (int p = 0; p < _system.rows(); Distance::invalidate_cache(p++))
        ;

    ++_accepted_steps;
    ++_total_steps;

    return _system;
}
