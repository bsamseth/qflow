#pragma once

#include "definitions.hpp"
#include "system.hpp"

#include <cstddef>

/**
 * Class implementing a general stochastic gradient decent algorithm.
 */
class SgdOptimizer
{
private:
    // Learning rate.
    Real _eta;

public:
    /**
     * Construct an SGD optimizer with a given learning rate.
     */
    explicit SgdOptimizer(Real eta = 0.1);

    /**
     * Return a parameter update meant to be added to the existing parameters
     * of the object/function that produced the given gradient.
     * @param gradient the gradient of the objective function to be minimized.
     * @return - _eta * gradient
     */
    virtual RowVector update_term(const RowVector& gradient);
};

/**
 * Class implementing the extension to SGD referred to as ADAM.
 */
class AdamOptimizer : public SgdOptimizer
{
private:
    const Real _alpha;
    Real       _alpha_t;
    const Real _beta1;
    const Real _beta2;
    const Real _epsilon;
    long       _t;
    RowVector  _m;
    RowVector  _v;

public:
    /**
     * Construct an ADAM optimizer with given parameters. The defaults are as proposed
     * in the original article.
     */
    AdamOptimizer(std::size_t n_parameters,
                  Real        alpha   = 0.001,
                  Real        beta1   = 0.9,
                  Real        beta2   = 0.999,
                  Real        epsilon = 1e-8);

    virtual RowVector update_term(const RowVector& gradient);
};
