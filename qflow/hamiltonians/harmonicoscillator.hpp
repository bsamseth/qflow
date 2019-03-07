#pragma once

#include "definitions.hpp"
#include "hamiltonian.hpp"
#include "system.hpp"
#include "wavefunction.hpp"

/**
 * Class representing the simple harmonic oscillator Hamiltonian.
 */
class HarmonicOscillator : public Hamiltonian
{
protected:
    const Real omega_ho_;
    const Real omega_z_;

public:
    explicit HarmonicOscillator(Real omega_ho = 1);

    HarmonicOscillator(Real omega_ho, Real omega_z, Real h = NUMMERIC_DIFF_STEP);

    Real external_potential(const System&) const override;

    /**
     * @return Zero, as no interaction is active.
     */
    Real internal_potential(const System&) const override;

    /**
     * Compute the Gross-Pitaevskii ideal case energy for a given system and
     * wavefunction.
     * @param system System to calculate for.
     * @param wavefunction Wavefunction to calculate for.
     * @return Gross-Pitaevskii ideal energy.
     */
    Real gross_pitaevskii_energy(const System& system,
                                 Wavefunction& wavefunction) const;
};

inline Real HarmonicOscillator::internal_potential(const System& system) const
{
    SUPPRESS_WARNING(system);
    return 0;
}

inline Real HarmonicOscillator::gross_pitaevskii_energy(const System& system,
                                                        Wavefunction& psi) const
{
    const int  N     = system.rows();
    const int  D     = system.cols();
    const Real alpha = psi.get_parameters()[0];
    const Real beta  = psi.get_parameters()[1];
    assert(beta == omega_z_);  // GP equation used is for when gamma = beta = omega_z.
    return N * (1 / (4 * alpha) + alpha) * (D == 3 ? 2 + beta : D) * 0.5;
}
