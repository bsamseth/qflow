#include "interactinghamiltonian.hpp"

#include "definitions.hpp"
#include "system.hpp"

#include <cassert>
#include <cmath>
#include <limits>

Real InteractingHamiltonian::local_energy(System& system, Wavefunction& psi) const
{
    // If internal potential is active ( > 0 means inf), no need to check more.
    if (internal_potential(system) > 0)
        return std::numeric_limits<Real>::max();

    return -0.5 * psi.laplacian(system) + external_potential(system);
}

Real InteractingHamiltonian::internal_potential(System& system) const
{
    for (int i = 0; i < system.rows() - 1; ++i)
    {
        for (int j = i + 1; j < system.rows(); ++j)
        {
            if (distance(system, i, j) <= _a)
                return std::numeric_limits<Real>::max();
        }
    }

    return 0.0;
}
