#pragma once

#include "definitions.hpp"
#include "boson.hpp"
#include "system.hpp"
#include "wavefunction.hpp"

class SimpleGaussian : public Wavefunction {
    public:
        // Inherit contructor.
        using Wavefunction::Wavefunction;

        virtual Real operator() (const System&) const;
        virtual Real derivative_alpha(const System&) const;
        virtual Real drift_force(const Boson&, int dim_index) const;
};

inline Real SimpleGaussian::drift_force(const Boson &boson, int dim_index) const {
    return -4 * _alpha * (dim_index == 2 ? _beta : 1) * boson[dim_index];
}
