#pragma once
#include <cassert>

#include "definitions.hpp"
#include "wavefunction.hpp"
#include "system.hpp"

class Hamiltonian {
    protected:
        Real _omega_z, _a, _h;

    public:

        Hamiltonian(Real omega_z = 1, Real a = 0, Real h = 0.001);

        virtual Real external_potential(const System&) const = 0;
        virtual Real internal_potential(const System&) const = 0;

        virtual Real local_energy(const System&, const Wavefunction&) const = 0;
        virtual Real local_energy_numeric(const System&, const Wavefunction&) const;
        virtual Real kinetic_energy(System &, const Wavefunction&) const;
        virtual Real derivative_alpha(const System&, const Wavefunction&) const = 0;
        Real gross_pitaevskii_energy(const System&, const Wavefunction&) const;

        friend std::ostream& operator<<(std::ostream&, const Hamiltonian&);
};

inline Real Hamiltonian::gross_pitaevskii_energy(const System &system, const Wavefunction &psi) const {
    const int N = system.get_n_bosons();
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

