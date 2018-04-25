#include <cmath>

#include "definitions.hpp"
#include "vector.hpp"
#include "system.hpp"
#include "simplegaussian.hpp"

namespace {
    Real exponent(const System &system, Real beta) {
        Real g = 0;
        if (system.get_dimensions() == 3) {
            for (const Vector &boson : system.get_particles()) {
                g += square(boson[0]) + square(boson[1]) + beta * square(boson[2]);
            }
        } else {
            for (const Vector &boson : system.get_particles()) {
                g += square(boson);
            }
        }
        return g;
    }
}

Real SimpleGaussian::operator() (System &system) const {
    return std::exp( - _alpha * exponent(system, _beta) );
}

Real SimpleGaussian::derivative_alpha(const System &system) const {
    return - exponent(system, _beta);
}

