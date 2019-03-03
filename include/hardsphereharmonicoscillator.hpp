#pragma once

#include "definitions.hpp"
#include "hamiltonian.hpp"
#include "harmonicoscillator.hpp"
#include "system.hpp"
#include "wavefunction.hpp"

/**
 * Class modelling an harmonic oscillator Hamiltonian with interactions.
 */
class HardSphereHarmonicOscillator : public HarmonicOscillator
{
private:
    Real a_;

public:
    using HarmonicOscillator::HarmonicOscillator;

    HardSphereHarmonicOscillator(Real omega_ho = 1,
                                 Real omega_z  = 1,
                                 Real a        = 0,
                                 Real h        = NUMMERIC_DIFF_STEP);

    Real internal_potential(const System&) const override;

    Real local_energy(const System&, Wavefunction&) const override;
};
