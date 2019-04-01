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
 */
class LennardJones : public Hamiltonian
{
public:
    static constexpr Real eps   = 10.22;  // K
    static constexpr Real sigma = 2.556;  // Å

    using Hamiltonian::Hamiltonian;

    /// No external potential defined in general, only internal.
    Real external_potential(const System&) const override;

    Real internal_potential(const System&) const override;
};

inline Real LennardJones::external_potential(const System& system) const
{
    SUPPRESS_WARNING(system);
    return 0;
}
