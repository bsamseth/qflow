#pragma once

#include "definitions.hpp"
#include "system.hpp"
#include "wavefunction.hpp"
#include "hamiltonian.hpp"

/**
 * Class representing the simple harmonic oscillator Hamiltonian.
 */
class HarmonicOscillatorHamiltonian: public Hamiltonian {
    public:

        using Hamiltonian::Hamiltonian;

        virtual Real external_potential(System&) const;

        /**
         * @return Zero, as no interaction is active.
         */
        virtual Real internal_potential(System&) const;
};

inline Real HarmonicOscillatorHamiltonian::internal_potential(__attribute__((unused)) System &system) const {
    return 0;
}
