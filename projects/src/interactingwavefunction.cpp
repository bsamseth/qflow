#include <cmath>
#include <cassert>

#include "definitions.hpp"
#include "vector.hpp"
#include "system.hpp"
#include "interactingwavefunction.hpp"

InteractingWavefunction::InteractingWavefunction(std::initializer_list<Real> parameters)
    : SimpleGaussian(parameters)
{
    // Set default parameters.
    const static Vector defaults = std::vector<Real>{{0.5, 1, 0}};
    _parameters = defaults;

    // Copy any given parameters.
    int i = 0;
    for (auto it = parameters.begin(); it != parameters.end() and i < defaults.size(); ++it, ++i)
        _parameters[i] = *it;
}

Real InteractingWavefunction::correlation(System &system) const {
    const auto a = _parameters[2];
    Real f = 1;
    for (int i = 0; i < system.get_n_particles() - 1; ++i) {
        for (int j = i + 1; j < system.get_n_particles(); ++j) {

            Real r_ij = system.distance(i, j);

            if (r_ij <= a)
                return 0.0;

            f *= 1 - a / r_ij;
        }
    }
    return f;
}

Real InteractingWavefunction::operator() (System &system) const {
    return correlation(system) * SimpleGaussian::operator()(system);
}

