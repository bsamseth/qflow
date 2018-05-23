#pragma once

#include "definitions.hpp"
#include "system.hpp"
#include "hamiltonian.hpp"
#include "rbmharmonicoscillatorhamiltonian.hpp"

/**
 * Class representing the hamiltonian with Coloumb interaction.
 */
class RBMInteractingHamiltonian : public RBMHarmonicOscillatorHamiltonian {
    public:

        using RBMHarmonicOscillatorHamiltonian::RBMHarmonicOscillatorHamiltonian;

        virtual Real internal_potential(System&) const;
};
