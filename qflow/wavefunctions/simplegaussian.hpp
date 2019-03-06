#pragma once

#include "definitions.hpp"
#include "system.hpp"
#include "vector.hpp"
#include "wavefunction.hpp"

/**
 * Class modelling a simple N-boson wavefunction w/o interaction.
 */
class SimpleGaussian : public Wavefunction
{
public:
    using Wavefunction::Wavefunction;
    SimpleGaussian(Real alpha = 0.5, Real beta = 1);

    /* SimpleGaussian(const RowVector& parameters); */

    Real operator()(const System&) override;

    RowVector gradient(const System& system) override;

    Real laplacian(const System& system) override;

    Real drift_force(const System& system, int k, int dim_index) override;

    Real derivative_alpha(const System& system) const;

    void set_parameters(const RowVector& parameters) override;
};

inline Real SimpleGaussian::drift_force(const System& system, int k, int dim_index)
{
    const auto alpha = _parameters[0];
    const auto beta  = _parameters[1];
    return -4 * alpha * (dim_index == 2 ? beta : 1) * system(k, dim_index);
}
