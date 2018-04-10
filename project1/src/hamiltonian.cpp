#include <iostream>
#include <limits>

#include "definitions.hpp"
#include "system.hpp"
#include "hamiltonian.hpp"

Hamiltonian::Hamiltonian(Real omega_z, Real a, Real h) : _omega_z(omega_z), _a(a), _h(h) {}

Real Hamiltonian::kinetic_energy(System &system, const Wavefunction &psi) const {
    Real E_k = -2 * (system.get_n_bosons() * system.get_dimensions()) * psi(system);

    for (int i = 0; i < system.get_n_bosons(); ++i) {
        for (int d = 0; d < system.get_dimensions(); ++d) {
            const auto temp = system[i][d];
            system(i, d) = temp + _h;
            E_k += psi(system);
            system(i, d) = temp - _h;
            E_k += psi(system);
            system(i, d) = temp;
        }
    }

    return -0.5 * E_k / (_h * _h);
}

Real Hamiltonian::local_energy_numeric(const System &const_system, const Wavefunction &psi) const {
    System &system = const_cast<System&>(const_system);
    Real wavefunc = psi(system);
    if (wavefunc == 0) {
        return std::numeric_limits<Real>::max();
    }

    return kinetic_energy(system, psi) / wavefunc
         + external_potential(system)
         + internal_potential(system);
}

