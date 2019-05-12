#pragma once

#include "definitions.hpp"
#include "system.hpp"
#include "vector.hpp"
#include "wavefunction.hpp"

/**
 * Class modelling a Jastrow factor on the following form:
 *
 *      J(X) = prod_{i < j} exp( - 1/2 (beta / r_iij)^n  )
 *
 * where r_ij is the distance between particle i and j. Variational
 * parameters are beta.
 *
 * Ref: W.L. McMillan, Phys. Rev. 138 (1965)
 */
class JastrowMcMillian : public Wavefunction
{
private:
    const int n_;

public:
    const Real L;

    JastrowMcMillian(int n, Real beta, Real L);

    Real operator()(const System&) override;

    RowVector gradient(const System& system) override;

    Real laplacian(const System& system) override;

    Real drift_force(const System& system, int k, int dim_index) override;
};
