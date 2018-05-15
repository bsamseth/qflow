#pragma once

#include <cstddef>
#include "definitions.hpp"
#include "system.hpp"
#include "wavefunction.hpp"
#include "hamiltonian.hpp"
#include "sampler.hpp"

namespace Optimizer {

/**
 * Return an optimized value for alpha.
 * @param wavefunction Wavefunction to calculate for.
 * @param hamiltonian Hamiltonian to calculate for.
 * @param sampler Source of sampled System instances.
 * @param initial_guess Starting point for alpha.
 * @param learning_rate Learning rate to use.
 * @param sample_points Number of samples to use per iteration.
 * @param max_iterations Maximum number of iterations to run.
 * @param dE_eps Gradient value for which to accept as zero.
 * @param verbose True if interim output should be printed.
 * @return Optimized value for alpha.
 */
Real gradient_decent_optimizer(Wavefunction &wavefunction,
                               const Hamiltonian &hamiltonian,
                               Sampler &sampler,
                               Real initial_guess,
                               Real learning_rate = 0.1,
                               int sample_points = 10000,
                               int max_iterations = 100,
                               Real dE_eps = 1e-8,
                               bool verbose = true);
}

class GeneralOptimizer {
    public:
        virtual Vector update_term(const Vector &gradient) = 0;
};

class SgdOptimizer : public GeneralOptimizer {
    private:
        Real _eta;

    public:
        SgdOptimizer(Real eta = 0.1);

        virtual Vector update_term(const Vector &gradient);
};


class AdamOptimizer : public GeneralOptimizer {
    private:
        const Real _alpha;
        Real _alpha_t;
        const Real _beta1;
        const Real _beta2;
        const Real _epsilon;
        long _t;
        Vector _m;
        Vector _v;

    public:
        AdamOptimizer(std::size_t n_parameters, Real alpha = 0.001, Real beta1 = 0.9, Real beta2 = 0.999, Real epsilon = 1e-8);

        virtual Vector update_term(const Vector &gradient);
};
