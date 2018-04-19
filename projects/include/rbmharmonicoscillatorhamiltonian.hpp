#pragma once

#include "definitions.hpp"
#include "system.hpp"
#include "rbmwavefunction.hpp"

/**
 * Class representing the simple harmonic oscillator Hamiltonian.
 */
class RBMHarmonicOscillatorHamiltonian {
    public:

        virtual Real external_potential(System&) const;

        /**
         * @return Zero, as no interaction is active.
         */
        virtual Real internal_potential(System&) const;

        virtual Real local_energy(System&, const RBMWavefunction&) const;
};

inline Real RBMHarmonicOscillatorHamiltonian::internal_potential(__attribute__((unused)) System &system) const {
    return 0;
}