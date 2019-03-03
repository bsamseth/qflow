#include "harmonicoscillator.hpp"

HarmonicOscillator::HarmonicOscillator(Real omega_ho)
    : HarmonicOscillator(omega_ho, omega_ho)
{
}

HarmonicOscillator::HarmonicOscillator(Real omega_ho, Real omega_z, Real h)
    : Hamiltonian(h), omega_ho_(omega_ho), omega_z_(omega_z)
{
}

Real HarmonicOscillator::external_potential(const System& system) const
{
    if (omega_z_ != omega_ho_ && system.cols() == 3)
    {
        Real pot = 0;
        for (int i = 0; i < system.rows(); ++i)
        {
            pot += square(omega_ho_) * (square(system(i, 0)) + square(system(i, 1)))
                   + square(omega_z_ * system(i, 2));
        }
        return 0.5 * pot;
    }
    else
    {
        return 0.5 * square(omega_ho_) * system.squaredNorm();
    }
}
