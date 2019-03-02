#pragma once

#include "definitions.hpp"
#include "hamiltonian.hpp"
#include "system.hpp"
#include "wavefunction.hpp"

/**
 * Class representing the simple harmonic oscillator Hamiltonian.
 */
class HarmonicOscillatorHamiltonian : public Hamiltonian
{
public:
    using Hamiltonian::Hamiltonian;

    Real external_potential(System&) const override;

    /**
     * @return Zero, as no interaction is active.
     */
    Real internal_potential(System&) const override;
};

inline Real HarmonicOscillatorHamiltonian::internal_potential(__attribute__((unused))
                                                              System& system) const
{
    return 0;
}
