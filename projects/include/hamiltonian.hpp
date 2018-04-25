#pragma once
#include <cassert>

#include "definitions.hpp"
#include "wavefunction.hpp"
#include "system.hpp"

/**
 * Class used to model a Hamiltonian.
 */
class Hamiltonian {
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
        virtual Real external_potential(System &system) const = 0;
        /**
         * Compute the interaction potential for a given system.
         * @param system System instance to calculate for.
         * @return Value of sum_{i < j} (V_int(r_i - r_j)).
         */
        virtual Real internal_potential(System &system) const = 0;

        /**
         * Compute the local energy for a given system and wavefunction.
         * @param system System to calculate for.
         * @param wavefunction Wavefunction to calculate for.
         * @return Local energy evaluation.
         */
        virtual Real local_energy(System &system, const Wavefunction &wavefunction) const = 0;
        /**
         * Compute the local energy for a given system and wavefunction, using numerical differentiation.
         * @param system System to calculate for.
         * @param wavefunction Wavefunction to calculate for.
         * @return Local energy evaluation.
         */
        virtual Real local_energy_numeric(System &system, const Wavefunction &wavefunction) const;
        /**
         * Compute the kinetic energy for a given system and wavefunction.
         * @param system System to calculate for.
         * @param wavefunction Wavefunction to calculate for.
         * @return Kinetic energy evaluation.
         */
        virtual Real kinetic_energy(System &system, const Wavefunction &wavefunction) const;
        /**
         * Compute the Gross-Pitaevskii ideal case energy for a given system and wavefunction.
         * @param system System to calculate for.
         * @param wavefunction Wavefunction to calculate for.
         * @return Gross-Pitaevskii ideal energy.
         */
        Real gross_pitaevskii_energy(const System &system, const Wavefunction &wavefunction) const;

        friend std::ostream& operator<<(std::ostream&, const Hamiltonian&);
};

inline Real Hamiltonian::gross_pitaevskii_energy(const System &system, const Wavefunction &psi) const {
    const int N = system.get_n_particles();
    const int D = system.get_dimensions();
    const Real alpha = psi.get_alpha();
    const Real beta = psi.get_beta();
    assert(beta == _omega_z);   // GP equation used is for when gamma = beta = omega_z.
    return N * (1 / (4 * alpha) + alpha) * (D == 3 ? 2 + beta : D) * 0.5;
}

inline std::ostream& operator<<(std::ostream &strm, const Hamiltonian &h) {
    return strm << "Hamiltonian(omega_z=" << h._omega_z
                << ", a=" << h._a << ")";
}

