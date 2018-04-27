#pragma once

#include <stdexcept>

#include "definitions.hpp"
#include "vector.hpp"
#include "system.hpp"
#include "simplegaussian.hpp"

/**
 * Class modeling the N-boson wavefunction with the Jastrow factor.
 */
class InteractingWavefunction : public SimpleGaussian {
    public:
        InteractingWavefunction(std::initializer_list<Real> parameters = {});

        virtual Real operator() (System&) const;
        virtual Real correlation(System&) const;
        virtual Real drift_force(const Vector&, int) const;
        virtual Real laplacian(System &system) const;
};

inline Real InteractingWavefunction::drift_force(const Vector &, int) const {
    throw std::logic_error("Importance sampling not implemented for interacting system.");
}

