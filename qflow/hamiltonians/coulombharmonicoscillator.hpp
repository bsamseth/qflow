#pragma once

#include "definitions.hpp"
#include "hamiltonian.hpp"
#include "harmonicoscillator.hpp"
#include "system.hpp"

/**
 * Class representing the Hamiltonian with Coulomb interaction.
 */
class CoulombHarmonicOscillator : public HarmonicOscillator
{
public:
    using HarmonicOscillator::HarmonicOscillator;

    Real internal_potential(const System&) const override;
};
