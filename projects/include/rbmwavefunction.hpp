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

class RBMWavefunction : public Wavefunction {

    protected:

        const Real _sigma2;
        const int _M;
        const int _N;

        Real u_i(int i, System &system) const;
        Real v_j(int j, System &system) const;

    public:

        RBMWavefunction(int M, int N, Real sigma2 = 1);

        virtual Vector gradient(System &system) const;

        virtual Real operator() (System &system) const;

        virtual Real deriv_a(int k, System &system) const;

        virtual Real deriv_b(int k, System &system) const;

        virtual Real deriv_w(int k, int l, System &system) const;

        virtual Real laplacian(System &system) const;

        virtual void train(const Hamiltonian &hamiltonian,
                           Sampler &sampler,
                           int iterations,
                           int sample_points,
                           Real learning_rate,
                           Real gamma = 0,
                           bool verbose = true);

        virtual Real drift_force(const Vector &, int) const;

        // Helpers. Public so that they are visible to the tests.
        int a(int i) const;
        int b(int j) const;
        int w(int i, int j) const;

};
inline Real RBMWavefunction::drift_force(const Vector &, int) const {
    throw std::logic_error("Function not implemented.");
}
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
