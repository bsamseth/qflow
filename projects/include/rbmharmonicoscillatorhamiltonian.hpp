#pragma once

#include "definitions.hpp"
#include "system.hpp"
#include "hamiltonian.hpp"
#include "wavefunction.hpp"

/**
 * Class representing the simple harmonic oscillator Hamiltonian.
 */
class RBMHarmonicOscillatorHamiltonian : public Hamiltonian {
    protected:
        Real _omega;

    public:

        RBMHarmonicOscillatorHamiltonian(Real omega = 1);

        virtual Real external_potential(System&) const;

        /**
         * @return Zero, as no interaction is active.
         */
        virtual Real internal_potential(System&) const;

        virtual Real local_energy(System&, const Wavefunction&) const;
};

inline Real RBMHarmonicOscillatorHamiltonian::internal_potential(__attribute__((unused)) System &system) const {
    return 0;
}
