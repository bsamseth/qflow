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

/**
 * Class for representing a wavefunction with an RBM.
 */
class RBMWavefunction : public Wavefunction {

    public:

        // Attributes set as public for easy access/modification.
        const Real _sigma2;  // The square of $\sigma$.
        const Real _root_factor;  // Factor used to change representation, psi -> sqrt(psi).
        const int _M;  // Number of visible nodes.
        const int _N;  // Number of hidden nodes.

        constexpr static Real GIBBS_FACTOR = 0.5;  // Setting _root_factor to this will result in
                                                   // the RBM suitable for Gibbs sampling.

        // The parameter vector is defined in the superclass.

        /**
         * Construct a wavefunction RBM.
         * @param M number of visible nodes.
         * @param N number of hidden nodes.
         * @param sigma2 square of the spread parameter $\sigma$.
         * @param root_factor default to 1, or set to `GIBBS_FACTOR` to get sqrt(psi).
         */
        RBMWavefunction(int M, int N, Real sigma2 = 1, Real root_factor = 1);

        /**
         * @param system configuration to evaluate the gradient for.
         * @return Gradient of the expected local energy, wrt. all the parameters.
         */
        RowVector gradient(System &system) const;

        /**
         * Evaluate the RBM.
         * @param system configuration to evaluate for.
         * @return Explicit value of the wavefunction, as given by the RBM.
         */
        Real operator() (System &system) const;

        /**
         * @param k bias index.
         * @param system configuration to evaluate for.
         * @return 1/psi d(psi)/d(a_k)
         */
        Real deriv_a(int k, System &system) const;

        /**
         * @param k bias index.
         * @param system configuration to evaluate for.
         * @return 1/psi d(psi)/d(b_k)
         */
        Real deriv_b(int k, System &system) const;

        /**
         * @param k first weight index.
         * @param l second weight index.
         * @param system configuration to evaluate for.
         * @return 1/psi d(psi)/d(w_kl)
         */
        Real deriv_w(int k, int l, System &system) const;

        /**
         * @param system configuration to evaluate for.
         * @return 1/psi laplacian(psi)
         */
        Real laplacian(System &system) const;

        /**
         * Train the parameters of the RBM in order to minimize the expected local energy.
         * @param hamiltonian the Hamiltonian used to evaluate the local energy.
         * @param sampler sampling algorithm used to produce new configurations.
         * @param iterations the number of training iterations to use.
         * @param sample_points the number of local energy evaluations to use per training iteration.
         * @param learning_rate hyper-parameter determining the magnitude of the updates given by SGD.
         * @param gamma hyper-parameter determining the amount of L2-regularization to use. Default is 0.
         * @param verbose if true extra output will be given at every iteration.
         */
        void train(const Hamiltonian &hamiltonian,
                Sampler &sampler,
                int iterations,
                int sample_points,
                Real learning_rate,
                Real gamma = 0,
                bool verbose = true);

        /**
         * Overload of previous train method, were the optimizer object is given directly.
         */
        void train(const Hamiltonian &hamiltonian,
                Sampler &sampler,
                int iterations,
                int sample_points,
                SgdOptimizer &optimizer,
                Real gamma = 0,
                bool verbose = true);


        /**
         * Return the drift force, as used in importance sampling.
         * @param system configuration to evaluate for.
         * @param particle_index which particle to evaluate the force for.
         * @param dim_index which dimension to evaluate the force for.
         * @return 2 grad(psi) / psi.
         */
        Real drift_force(const System &system, int particle_index, int dim_index) const;

        // Helpers. Public so that they are visible to the tests.
        Real v_j(int j, const System &system) const;
        // The following functions return the index into the full parameter vector which
        // point to the specified parameter.
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
