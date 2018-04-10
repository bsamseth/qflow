#include <cmath>
#include <cassert>

#include "definitions.hpp"
#include "boson.hpp"
#include "system.hpp"
#include "interactingwavefunction.hpp"

Real InteractingWavefunction::correlation(System &system) const {
    Real f = 1;
    for (int i = 0; i < system.get_n_bosons() - 1; ++i) {
        for (int j = i + 1; j < system.get_n_bosons(); ++j) {

            Real r_ij = system.distance(i, j);

            if (r_ij <= _a)
                return 0.0;

            f *= 1 - _a / r_ij;
        }
    }
    return f;
}

Real InteractingWavefunction::operator() (System &system) const {
    return correlation(system) * SimpleGaussian::operator()(system);
}

