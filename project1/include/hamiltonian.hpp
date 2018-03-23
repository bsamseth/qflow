#pragma once

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
        virtual Real local_energy_numeric(System&, const Wavefunction&) const;
        virtual Real kinetic_energy(System &, const Wavefunction&) const;
        virtual Real derivative_alpha(const System&, const Wavefunction&) const = 0;

        friend std::ostream& operator<<(std::ostream&, const Hamiltonian&);
};

inline std::ostream& operator<<(std::ostream &strm, const Hamiltonian &h) {
    return strm << "Hamiltonian(omega_z=" << h._omega_z
                << ", a=" << h._a << ")";
}

