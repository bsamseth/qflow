#pragma once

#include "definitions.hpp"
#include "system.hpp"
#include "vector.hpp"
#include "wavefunction.hpp"

/**
 * Class modelling a Jastrow factor on the following form:
 *
 *      J(X) = prod_{i < j} exp( - 1/2 (beta r_ij)^2 + |beta gamma| r_ij )
 *
 * where r_ij is the distance between particle i and j. Variational
 * parameters are beta and gamma.
 *
 * Ref: Ciftja, Orion. (2009). a Jastrow Correlation Factor for Two-Dimensional
 *      Parabolic Quantum Dots. Modern Physics Letters B - MOD PHYS LETT B. 23.
 *      3055-3064. 10.1142/S0217984909021120.
 */
class JastrowOrion : public Wavefunction
{
public:
    JastrowOrion(Real beta = 1, Real gamma = 0);

    Real operator()(const System&) override;

    RowVector gradient(const System& system) override;

    Real laplacian(const System& system) override;

    Real drift_force(const System& system, int k, int dim_index) override;
};
