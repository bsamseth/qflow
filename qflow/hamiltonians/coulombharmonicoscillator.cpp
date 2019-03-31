#include "coulombharmonicoscillator.hpp"

#include "definitions.hpp"
#include "distance.hpp"
#include "system.hpp"

Real CoulombHarmonicOscillator::internal_potential(const System& system) const
{
    const auto P   = system.rows();
    Real       pot = 0;
    for (int i = 0; i < P - 1; ++i)
    {
        for (int j = i + 1; j < P; ++j)
        {
            pot += 1.0 / Distance::probe(system, i, j);
        }
    }
    return pot;
}
