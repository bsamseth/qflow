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
        // Inherit contructor.
        using SimpleGaussian::SimpleGaussian;

        virtual Real operator() (System&) const;
        virtual Real correlation(System&) const;
        virtual Real drift_force(const Vector&, int) const;
};

inline Real InteractingWavefunction::drift_force(const Vector &, int) const {
    throw std::logic_error("Importance sampling not implemented for interacting system.");
}

