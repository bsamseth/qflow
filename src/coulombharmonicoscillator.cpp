#include "definitions.hpp"
#include "rbminteractinghamiltonian.hpp"
#include "system.hpp"
RBMInteractingHamiltonian::RBMInteractingHamiltonian(Real omega)
    : RBMHarmonicOscillatorHamiltonian(omega)
{
}

Real RBMInteractingHamiltonian::internal_potential(System& system) const
{
    const auto P   = system.rows();
    Real       pot = 0;
    for (int i = 0; i < P - 1; ++i)
    {
        for (int j = i + 1; j < P; ++j)
        {
            pot += 1.0 / distance(system, i, j);
        }
    }
    return pot;
}
