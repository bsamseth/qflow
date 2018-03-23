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
        virtual Boson drift_force(const System&, int boson) const;
};
