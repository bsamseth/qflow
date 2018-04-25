#pragma once

#include "definitions.hpp"
#include "vector.hpp"
#include "system.hpp"
#include "wavefunction.hpp"

/**
 * Class modelling a simple N-boson wavefunction w/o interaction.
 */
class SimpleGaussian : public Wavefunction {
    public:
        // Inherit contructor.
        using Wavefunction::Wavefunction;

        virtual Real operator() (System&) const;
        virtual Real derivative_alpha(const System&) const;
        virtual Real drift_force(const Vector&, int dim_index) const;
};

inline Real SimpleGaussian::drift_force(const Vector &boson, int dim_index) const {
    return -4 * _alpha * (dim_index == 2 ? _beta : 1) * boson[dim_index];
}
