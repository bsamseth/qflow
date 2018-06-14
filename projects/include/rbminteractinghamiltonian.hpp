#pragma once

#include "definitions.hpp"
#include "system.hpp"
#include "hamiltonian.hpp"
#include "rbmharmonicoscillatorhamiltonian.hpp"

/**
 * Class representing the Hamiltonian with Coulomb interaction.
 */
class RBMInteractingHamiltonian : public RBMHarmonicOscillatorHamiltonian {
    public:

        RBMInteractingHamiltonian(Real omega = 1.0);

        virtual Real internal_potential(System&) const;
};
