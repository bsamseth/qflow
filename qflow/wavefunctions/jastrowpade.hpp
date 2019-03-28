#pragma once

#include "definitions.hpp"
#include "system.hpp"
#include "vector.hpp"
#include "wavefunction.hpp"

/**
 * Class modelling a Jastrow factor on the following form:
 *
 *      J(X) = prod_{i < j} exp(  alpha r_ij / ( 1 + beta r_ij )  )
 *
 * where r_ij is the distance between particle i and j. Variational
 * parameters are beta, and alpha is fixed as 1/2 for opposite spins,
 * and 1/4 for equal spins.
 */
class JastrowPade : public Wavefunction
{
private:
    const Real alpha_;

public:
    JastrowPade(Real alpha = 0.5, Real beta = 1);

    Real operator()(const System&) override;

    RowVector gradient(const System& system) override;

    Real laplacian(const System& system) override;

    Real drift_force(const System& system, int k, int dim_index) override;
};
