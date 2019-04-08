#pragma once

#include "definitions.hpp"
#include "hamiltonian.hpp"
#include "system.hpp"
#include "wavefunction.hpp"

/**
 * Class representing the HFDHE2 effective potential for Helium 4:
 *
 *     V(r) = eps * (
 *                    A * exp(-alpha r / r_m)
 *                  - F(r) [
 *                      C_6  (r_m / r)^6
 *                    + C_8  (r_m / r)^8
 *                    + C_10 (r_m / r)^10
 *                    ]
 *                  )
 * with
 *
 *             _
 *            |   exp( - [D r_m / r   -  1]^2 )   for r/r_m <= D
 *    F(r) = -|
 *            |_  1                               otherwise
 *
 * where r is a inter-atomic distance r = |r_i - r_j|.
 *
 * There are no free parameters, and the values for all are as follows:
 *
 *             A     =  0.5448504 * 10^6     eps / kappa  =  10.8 K
 *             alpha =  13.353384            C_6          =  1.37732412
 *             D     =  1.241314             C_8          =  0.4253785
 *             r_m   =  2.9673 Å             C_10         =  0.178100
 *
 *
 * Ref: Aziz et al, Chem. Phys 70, 4330 (1979).
 * Condensed ref: Ruggeri et al, doi: 10.1103/physrevlett.120.205302
 */
class HFDHE2 : public Hamiltonian
{
public:
    static constexpr Real A     = 544850.4;
    static constexpr Real alpha = 13.353384;
    static constexpr Real D     = 1.241314;
    static constexpr Real r_m   = 2.9673;  // Å
    static constexpr Real eps   = 10.8;    // K
    static constexpr Real C_6   = 1.37732412;
    static constexpr Real C_8   = 0.4253785;
    static constexpr Real C_10  = 0.178100;

    using Hamiltonian::Hamiltonian;

    /// No external potential defined in general, only internal.
    Real external_potential(const System&) const override;

    Real internal_potential(const System&) const override;
};

inline Real HFDHE2::external_potential(const System& system) const
{
    SUPPRESS_WARNING(system);
    return 0;
}
