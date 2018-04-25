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
    protected:
        Real _a = 0;

    public:
        InteractingWavefunction(std::initializer_list<Real> parameters = {});

        virtual Real operator() (System&) const;
        virtual Real correlation(System&) const;
        virtual Real drift_force(const Vector&, int) const;
};

inline Real InteractingWavefunction::drift_force(const Vector &, int) const {
    throw std::logic_error("Importance sampling not implemented for interacting system.");
}

