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

/**
 * Class implementing a general stochastic gradient decent algorithm.
 */
class SgdOptimizer {
    private:
        // Learning rate.
        Real _eta;

    public:
        /**
         * Construct an SGD optimizer with a given learning rate.
         */
        SgdOptimizer(Real eta = 0.1);

        /**
         * Return a parameter update meant to be added to the existing parameters
         * of the object/function that produced the given gradient.
         * @param gradient the gradient of the objective function to be minimized.
         * @return - _eta * gradient
         */
        virtual Vector update_term(const Vector &gradient);
};


/**
 * Class implementing the extension to SGD referred to as ADAM.
 */
class AdamOptimizer : public SgdOptimizer {
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
        /**
         * Construct an ADAM optimizer with given parameters. The defaults are as proposed in the original article.
         */
        AdamOptimizer(std::size_t n_parameters, Real alpha = 0.001, Real beta1 = 0.9, Real beta2 = 0.999, Real epsilon = 1e-8);

        virtual Vector update_term(const Vector &gradient);
};
