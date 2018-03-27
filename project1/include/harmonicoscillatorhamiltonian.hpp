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

        virtual Real external_potential(const System&) const;

        /**
         * @return Zero, as no interaction is active.
         */
        virtual Real internal_potential(const System&) const;

        virtual Real local_energy(const System&, const Wavefunction&) const;
};

inline Real HarmonicOscillatorHamiltonian::internal_potential(__attribute__((unused)) const System &system) const {
    return 0;
}
