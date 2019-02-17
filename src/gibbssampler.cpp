#include <cmath>
#include "definitions.hpp"
#include "system.hpp"
#include "wavefunction.hpp"
#include "hamiltonian.hpp"
#include "gibbssampler.hpp"



GibbsSampler::GibbsSampler(const System &system, RBMWavefunction &wavefunction, std::size_t N)
    : Sampler(system, wavefunction, 0, N), _rbm(wavefunction)
{
    for (std::size_t i = 0; i < _N_instances; ++i) {
        initialize_system(i);
    }
}

void GibbsSampler::initialize_system(std::size_t i) {
    StateInfo& state = _instances[i];
    _stddev = std::sqrt(_rbm._sigma2);
    std::normal_distribution<Real> dist(0, _stddev);
    for (int i = 0; i < state.system_old.rows(); ++i) {
        for (int d = 0; d < state.system_old.cols(); ++d) {
            state.system_old(i, d) = dist(rand_gen);
        }
    }
}

System& GibbsSampler::next_configuration(std::size_t i) {
    StateInfo& state = _instances[i];
    RowVector h (_rbm._N);
    for (int j = 0; j < _rbm._N; ++j) {
        h[j] = unif(rand_gen) < P_h(j, state.system_old) ? 1 : 0;
    }

    for (int i = 0; i < _rbm._M; ++i) {
        Real mean = _rbm.get_parameters()[_rbm.a(i)];
        for (int j = 0; j < _rbm._N; ++j) {
            mean += _rbm.get_parameters()[_rbm.w(i, j)] * h[j];
        }

        std::normal_distribution<Real> dist(mean, _stddev);
        state.system_old.data()[i] = dist(rand_gen);
    }

    ++state.accepted_steps;
    ++state.total_steps;

    return state.system_old;
}
