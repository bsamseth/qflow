#pragma once

#include "definitions.hpp"
#include "hamiltonian.hpp"
#include "harmonicoscillatorhamiltonian.hpp"
#include "system.hpp"
#include "wavefunction.hpp"

/**
 * Class modelling an harmonic oscillator Hamiltonian with interactions.
 */
class InteractingHamiltonian : public HarmonicOscillatorHamiltonian
{
public:
    using HarmonicOscillatorHamiltonian::HarmonicOscillatorHamiltonian;

    Real internal_potential(System&) const override;

    Real local_energy(System&, Wavefunction&) const override;
};
