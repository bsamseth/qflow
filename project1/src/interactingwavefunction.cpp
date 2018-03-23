#include <cmath>
#include <cassert>

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

Boson InteractingWavefunction::drift_force(const System &system, int k) const {
    // Reimplement first part so as to save a copy operation.
    const Boson &r_k = system[k];
    Boson F = system[k];
    if (system.get_dimensions() == 3) {
        F[2] *= _beta;
    }

    F *= -2 * _alpha;

    // Interaction contribution.
    for (int j = 0; j < system.get_n_bosons(); ++j) {
        if (j == k) continue;

        Boson r_kj = r_k - system[j];
        Real r_kj_norm = std::sqrt( square( r_kj ) );

        // Should not be in a state where two particles are on top of each other.
        // Should only maybe happen with unlucky initialization. Assert to be safe,
        // if it happens at least we will know (when debugging).
        assert(r_kj_norm > 0);

        // Inplace operations saves copies.
        r_kj *= _a / (square(r_kj_norm) * (r_kj_norm - _a));
        F += r_kj;
    }

    F *= 2;

    return F;
}
