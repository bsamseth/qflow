#pragma once

#include <stdexcept>

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
        virtual Real drift_force(const Boson&, int) const;
};

inline Real InteractingWavefunction::drift_force(const Boson &, int) const {
    throw std::logic_error("Importance sampling not implemented for interacting system.");
}

