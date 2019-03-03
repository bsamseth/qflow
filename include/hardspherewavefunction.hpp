#pragma once

#include "definitions.hpp"
#include "simplegaussian.hpp"
#include "system.hpp"
#include "vector.hpp"

#include <stdexcept>

/**
 * Class modeling the N-boson wavefunction with the Jastrow factor.
 */
class HardSphereWavefunction : public SimpleGaussian
{
public:
    HardSphereWavefunction(Real alpha = 0.5, Real beta = 1, Real a = 0);

    Real operator()(const System&) override;
    Real correlation(const System&) const;
    Real drift_force(const System&, int, int) override;
    Real laplacian(const System& system) override;
};

inline Real HardSphereWavefunction::drift_force(const System&, int, int)
{
    throw std::logic_error(
        "Importance sampling not implemented for interacting system.");
}
