#include "definitions.hpp"
#include "system.hpp"
#include "harmonicoscillatorhamiltonian.hpp"

Real HarmonicOscillatorHamiltonian::external_potential(System &system) const {
    Real pot = 0;

    for (int i = 0; i < system.cols(); ++i) {
        if (system.rows() == 3) {
            pot += square(system(0, i)) + square(system(1, i)) + square(_omega_z * system(2, i));
        } else {
            pot += squaredNorm(system.col(i));
        }
    }

    return 0.5 * pot;
}

