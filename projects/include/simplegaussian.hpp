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

        SimpleGaussian(std::initializer_list<Real> parameters = {});

        virtual Real operator() (System&) const;

        virtual Vector gradient(System &system) const;

        virtual Real laplacian(System &system) const;

        Real drift_force(const Vector &boson, int dim_index) const;

        Real derivative_alpha(const System &system) const;

};

inline Real SimpleGaussian::drift_force(const Vector &boson, int dim_index) const {
    const auto alpha = _parameters[0];
    const auto beta  = _parameters[1];
    return -4 * alpha * (dim_index == 2 ? beta : 1) * boson[dim_index];
}
