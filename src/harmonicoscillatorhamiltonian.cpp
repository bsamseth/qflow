#include "harmonicoscillatorhamiltonian.hpp"

#include "definitions.hpp"
#include "system.hpp"

Real HarmonicOscillatorHamiltonian::external_potential(System& system) const
{
    Real pot = 0;

    for (int i = 0; i < system.rows(); ++i)
    {
        if (system.cols() == 3)
        {
            pot += square(system(i, 0)) + square(system(i, 1))
                   + square(_omega_z * system(i, 2));
        }
        else
        {
            pot += squaredNorm(system.row(i));
        }
    }

    return 0.5 * pot;
}
