#pragma once

#include "definitions.hpp"
#include "boson.hpp"
#include "system.hpp"
#include "simplegaussian.hpp"

class InteractingWavefunction : public SimpleGaussian {
    public:
        // Inherit contructor.
        using SimpleGaussian::SimpleGaussian;

        virtual Real operator() (const System&) const;
        virtual Real derivative_alpha(const System&) const;
        virtual Real correlation(const System&) const;
        virtual Boson drift_force(const System&, int boson) const;
};
