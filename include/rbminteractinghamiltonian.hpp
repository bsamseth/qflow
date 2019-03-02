#pragma once

#include "definitions.hpp"
#include "hamiltonian.hpp"
#include "rbmharmonicoscillatorhamiltonian.hpp"
#include "system.hpp"

/**
 * Class representing the Hamiltonian with Coulomb interaction.
 */
class RBMInteractingHamiltonian : public RBMHarmonicOscillatorHamiltonian
{
public:
    RBMInteractingHamiltonian(Real omega = 1.0);

    Real internal_potential(System&) const override;
};
