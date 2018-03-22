#include <cmath>

#include "definitions.hpp"
#include "boson.hpp"
#include "system.hpp"
#include "interactingwavefunction.hpp"

Real InteractingWavefunction::correlation(const System &system) const {
    Real f = 1;
    for (int i = 0; i < system.get_n_bosons() - 1; ++i) {
        const Boson &r_i = system[i];
        for (int j = i + 1; j < system.get_n_bosons(); ++j) {

            Real r_ij = std::sqrt( square(r_i - system[j]) );

            if (r_ij <= _a)
                return 0.0;

            f *= 1 - _a / r_ij;
        }
    }
    return f;
}

Real InteractingWavefunction::operator() (const System &system) const {
    return correlation(system) * SimpleGaussian::operator()(system);
}

Real InteractingWavefunction::derivative_alpha(const System &system) const {
    return correlation(system) * SimpleGaussian::derivative_alpha(system);
}


