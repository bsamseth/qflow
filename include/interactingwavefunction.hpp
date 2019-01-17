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
        InteractingWavefunction(Real alpha = 0.5, Real beta = 1, Real a = 0);

        virtual Real operator() (System&);
        virtual Real correlation(System&) const;
        virtual Real drift_force(const System&, int, int);
        virtual Real laplacian(System &system);
};

inline Real InteractingWavefunction::drift_force(const System &, int, int) {
    throw std::logic_error("Importance sampling not implemented for interacting system.");
}

