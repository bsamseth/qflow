#pragma once
#include "definitions.hpp"
#include "optimizer.hpp"
#include "sampler.hpp"
#include "system.hpp"
#include "wavefunction.hpp"

#include <cassert>

/**
 * Class used to model a Hamiltonian.
 */
class Hamiltonian
{
protected:
    Real h_;

public:
    /**
     * Instantiate a Hamiltonian with the given parameters.
     * @param h Step to use in numerical differentiation.
     */
    explicit Hamiltonian(Real h = NUMMERIC_DIFF_STEP);

    /**
     * Compute the oscillator trap potential for a given system.
     * @param system System instance to calculate for.
     * @return Value of sum_i (V_ext(r_i)).
     */
    virtual Real external_potential(const System& system) const = 0;
    /**
     * Compute the interaction potential for a given system.
     * @param system System instance to calculate for.
     * @return Value of sum_{i < j} (V_int(r_i - r_j)).
     */
    virtual Real internal_potential(const System& system) const = 0;

    /**
     * Compute the local energy for a given system and wavefunction.
     * @param system System to calculate for.
     * @param wavefunction Wavefunction to calculate for.
     * @return Local energy evaluation.
     */
    virtual Real local_energy(const System& system, Wavefunction& wavefunction) const;
    /**
     * Compute the local energy for a given system and wavefunction, using numerical
     * differentiation.
     * @param system System to calculate for.
     * @param wavefunction Wavefunction to calculate for.
     * @return Local energy evaluation.
     */
    virtual Real local_energy_numeric(const System& system,
                                      Wavefunction& wavefunction) const;
    /**
     * Compute the kinetic energy for a given system and wavefunction.
     * @param system System to calculate for.
     * @param wavefunction Wavefunction to calculate for.
     * @return Kinetic energy evaluation.
     */
    virtual Real kinetic_energy(const System& system, Wavefunction& wavefunction) const;
    /**
     * Compute the kinetic energy for a given system and wavefunction, using numerical
     * differentiation.
     * @param system System to calculate for.
     * @param wavefunction Wavefunction to calculate for.
     * @return Kinetic energy evaluation.
     */
    virtual Real kinetic_energy_numeric(const System& system,
                                        Wavefunction& wavefunction) const;

    RowVector
        local_energy_gradient(Sampler& sampler, Wavefunction& psi, long samples) const;

    Real local_energy(Sampler& sampler, Wavefunction& psi, long samples) const;

    Real mean_distance(Sampler&, long samples) const;

    RowVector onebodydensity(Sampler& sampler,
                             int      n_bins,
                             Real     max_radius,
                             long     samples) const;

    void optimize_wavefunction(Wavefunction& psi,
                               Sampler&      sampler,
                               int           iterations,
                               int           sample_points,
                               SgdOptimizer& optimizer,
                               Real          gamma,
                               bool          verbose) const;
};