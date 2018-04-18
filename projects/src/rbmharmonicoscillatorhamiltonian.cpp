
#include "rbmharmonicoscillatorhamiltonian.hpp"



Real RBMHarmonicOscillatorHamiltonian::external_potential(System &system) const {
    Real res = 0;
    for (auto &&particle : system.get_bosons()) {
        res += square(particle);
    }
    return 0.5 * res;
}

Real RBMHarmonicOscillatorHamiltonian::local_energy(System &system, const RBMWavefunction &psi) const {
    return external_potential(system) + internal_potential(system) - 0.5 * psi.laplacian(system);
}
