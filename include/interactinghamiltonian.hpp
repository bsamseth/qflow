#pragma once

#include "definitions.hpp"
#include "system.hpp"
#include "wavefunction.hpp"
#include "hamiltonian.hpp"
#include "harmonicoscillatorhamiltonian.hpp"

/**
 * Class modelling an harmonic oscillator Hamiltonian with interactions.
 */
class InteractingHamiltonian: public HarmonicOscillatorHamiltonian {
    public:

        using HarmonicOscillatorHamiltonian::HarmonicOscillatorHamiltonian;

        virtual Real internal_potential(System&) const;

        virtual Real local_energy(System &, Wavefunction&) const;
};
