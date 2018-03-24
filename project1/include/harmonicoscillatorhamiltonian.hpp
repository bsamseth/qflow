#pragma once

#include "definitions.hpp"
#include "system.hpp"
#include "wavefunction.hpp"
#include "hamiltonian.hpp"

class HarmonicOscillatorHamiltonian: public Hamiltonian {
    public:

        using Hamiltonian::Hamiltonian;

        virtual Real external_potential(const System&) const;

        virtual Real internal_potential(const System&) const;

        virtual Real local_energy(const System&, const Wavefunction&) const;

        virtual Real derivative_alpha(const System&, const Wavefunction&) const;
};

inline Real HarmonicOscillatorHamiltonian::internal_potential(__attribute__((unused)) const System &system) const {
    return 0;
}
