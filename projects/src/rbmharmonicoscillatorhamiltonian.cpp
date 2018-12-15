
#include "rbmharmonicoscillatorhamiltonian.hpp"


RBMHarmonicOscillatorHamiltonian::RBMHarmonicOscillatorHamiltonian(Real omega) : _omega(omega) {}

Real RBMHarmonicOscillatorHamiltonian::external_potential(System &system) const {
    Real res = square(system).sum();
    return 0.5 * square(_omega) * res;
}

Real RBMHarmonicOscillatorHamiltonian::local_energy(System &system, const Wavefunction &psi) const {
    return external_potential(system) + internal_potential(system) - 0.5 * psi.laplacian(system);
}
