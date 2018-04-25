#pragma once

#include "definitions.hpp"
#include "vector.hpp"
#include "system.hpp"
#include "wavefunction.hpp"

/**
 * Class modelling a simple N-boson wavefunction w/o interaction.
 */
class SimpleGaussian : public Wavefunction {
     protected:
        Real _alpha = 0.5, _beta = 1;

    public:

        SimpleGaussian(std::initializer_list<Real> parameters = {});

        virtual Real operator() (System&) const;

        virtual Vector gradient(System &system) const;

        virtual Real laplacian(System &system) const;

        Real drift_force(const Vector &boson, int dim_index) const;

        Real derivative_alpha(const System &system) const;

};

inline Real SimpleGaussian::drift_force(const Vector &boson, int dim_index) const {
    return -4 * _alpha * (dim_index == 2 ? _beta : 1) * boson[dim_index];
}
