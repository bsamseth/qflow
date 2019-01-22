#include "definitions.hpp"
#include "wavefunction.hpp"

Wavefunction::Wavefunction(const RowVector& parameters)
    : _parameters(parameters) {}


RowVector Wavefunction::drift_force(const System& system) {
    const int n_particles = system.rows();
    const int n_dimensions = system.cols();
    RowVector force(system.size());

    for (int k = 0; k < n_particles; ++k)
        for (int d = 0; d < n_dimensions; ++d)
            force(n_dimensions * k + d) = drift_force(system, k, d);

    return force;
}
