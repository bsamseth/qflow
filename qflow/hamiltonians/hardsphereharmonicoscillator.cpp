#include "hardsphereharmonicoscillator.hpp"

#include "definitions.hpp"
#include "distance.hpp"
#include "system.hpp"

#include <cassert>
#include <cmath>
#include <limits>

HardSphereHarmonicOscillator::HardSphereHarmonicOscillator(Real omega_ho,
                                                           Real omega_z,
                                                           Real a,
                                                           Real h)
    : HarmonicOscillator(omega_ho, omega_z, h), a_(a)
{
}

Real HardSphereHarmonicOscillator::local_energy(const System& system,
                                                Wavefunction& psi) const
{
    // If internal potential is active ( > 0 means inf), no need to check more.
    if (internal_potential(system) > 0)
        return std::numeric_limits<Real>::max();

    return -0.5 * psi.laplacian(system) + external_potential(system);
}

Real HardSphereHarmonicOscillator::internal_potential(const System& system) const
{
    for (int i = 0; i < system.rows() - 1; ++i)
    {
        for (int j = i + 1; j < system.rows(); ++j)
        {
            if (Distance::probe(system, i, j) <= a_)
                return std::numeric_limits<Real>::max();
        }
    }
    return 0.0;
}
