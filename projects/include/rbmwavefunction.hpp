#pragma once

#include <vector>
#include <algorithm>
#include <functional>
#include <cassert>

#include "definitions.hpp"
#include "system.hpp"
#include "wavefunction.hpp"
#include "sampler.hpp"
#include "hamiltonian.hpp"
#include "optimizer.hpp"

class RBMWavefunction : public Wavefunction {

    public:

        const Real _sigma2;
        const Real _root_factor;
        const int _M;
        const int _N;

        constexpr static Real GIBBS_FACTOR = 0.5;

        RBMWavefunction(int M, int N, Real sigma2 = 1, Real root_factor = 1);

        Vector gradient(System &system) const;

        Real operator() (System &system) const;

        Real deriv_a(int k, System &system) const;

        Real deriv_b(int k, System &system) const;

        Real deriv_w(int k, int l, System &system) const;

        Real laplacian(System &system) const;

        void train(const Hamiltonian &hamiltonian,
                Sampler &sampler,
                int iterations,
                int sample_points,
                Real learning_rate,
                Real gamma = 0,
                bool verbose = true);

        void train(const Hamiltonian &hamiltonian,
                Sampler &sampler,
                int iterations,
                int sample_points,
                SgdOptimizer &optimizer,
                Real gamma = 0,
                bool verbose = true);


        Real drift_force(const System &system, int particle_index, int) const;

        // Helpers. Public so that they are visible to the tests.
        Real v_j(int j, const System &system) const;
        int a(int i) const;
        int b(int j) const;
        int w(int i, int j) const;

};

inline int RBMWavefunction::a(int i) const {
    assert(i >= 0 and i < _M);
    return i;
}
inline int RBMWavefunction::b(int j) const {
    assert(j >= 0 and j < _N);
    return _M + j;
}
inline int RBMWavefunction::w(int i, int j) const {
    assert(i >= 0 and i < _M);
    assert(j >= 0 and j < _N);
    return _M + _N + i * _N + j;
}
