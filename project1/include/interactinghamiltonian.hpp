#pragma once

#include "definitions.hpp"
#include "system.hpp"
#include "wavefunction.hpp"
#include "hamiltonian.hpp"
#include "harmonicoscillatorhamiltonian.hpp"

class InteractingHamiltonian: public HarmonicOscillatorHamiltonian {
    public:

        using HarmonicOscillatorHamiltonian::HarmonicOscillatorHamiltonian;

        virtual Real internal_potential(const System&) const;

        virtual Real local_energy(const System&, const Wavefunction&) const;
};
