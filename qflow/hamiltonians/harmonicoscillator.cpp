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
        static Vector coeffs(3);
        coeffs << square(omega_ho_), square(omega_ho_), square(omega_z_);
        return 0.5 * (system.array().square().matrix() * coeffs).sum();
    }
    else
    {
        return 0.5 * square(omega_ho_) * system.squaredNorm();
    }
}
