#pragma once

#include "definitions.hpp"
#include "hamiltonian.hpp"
#include "system.hpp"
#include "wavefunction.hpp"

/**
 * Class representing the Lennard-Jones effective potential for Helium 4:
 *
 *     V(r) = 4 * eps * ( (sigma / r)^12 - (sigma / r)^6 )
 *
 * where r is a inter-atomic distance r = |r_i - r_j|.
 *
 * There are no free parameters, and the values for eps and sigma are:
 *
 *             eps / kappa  =  10.22 K
 *             sigma        =  2.556 Å
 *
 * Also using the following value:
 *             hbar^2 / m_helium k_B = 12.1193048 Å^2 K
 */
class LennardJones : public Hamiltonian
{
public:
    static constexpr Real hbar2_per_m = 12.1193048;
    static constexpr Real eps         = 10.22;  // K
    static constexpr Real sigma       = 2.556;  // Å
    static constexpr Real r_core      = 0.3 * sigma;
    const Real            L;
    Real                  truncation_potential;

    explicit LennardJones(Real L, Real h = NUMMERIC_DIFF_STEP);

    /// No external potential defined in general, only internal.
    Real external_potential(const System&) const override;

    Real internal_potential(const System&) const override;

private:
    Real vlj_core(Real r2) const;
};

inline Real LennardJones::external_potential(const System& system) const
{
    SUPPRESS_WARNING(system);
    return 0;
}
