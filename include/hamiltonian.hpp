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
    Real _omega_z, _a, _h;

public:
    /**
     * Instantiate a Hamiltonian with the given parameters.
     * @param omega_z Z-component of the oscillator trap.
     * @param a Boson hard-sphere diameter.
     * @param h Step to use in numerical differentiation.
     */
    Hamiltonian(Real omega_z = 1, Real a = 0, Real h = 0.001);

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

    /**
     * Compute the Gross-Pitaevskii ideal case energy for a given system and
     * wavefunction.
     * @param system System to calculate for.
     * @param wavefunction Wavefunction to calculate for.
     * @return Gross-Pitaevskii ideal energy.
     */
    Real gross_pitaevskii_energy(const System& system,
                                 Wavefunction& wavefunction) const;

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

    friend std::ostream& operator<<(std::ostream&, const Hamiltonian&);
};

inline Real Hamiltonian::gross_pitaevskii_energy(const System& system,
                                                 Wavefunction& psi) const
{
    const int  N     = system.rows();
    const int  D     = system.cols();
    const Real alpha = psi.get_parameters()[0];
    const Real beta  = psi.get_parameters()[1];
    assert(beta == _omega_z);  // GP equation used is for when gamma = beta = omega_z.
    return N * (1 / (4 * alpha) + alpha) * (D == 3 ? 2 + beta : D) * 0.5;
}

inline std::ostream& operator<<(std::ostream& strm, const Hamiltonian& h)
{
    return strm << "Hamiltonian(omega_z=" << h._omega_z << ", a=" << h._a << ")";
}
