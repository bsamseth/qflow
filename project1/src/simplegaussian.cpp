#include <cmath>

#include "definitions.hpp"
#include "boson.hpp"
#include "system.hpp"
#include "simplegaussian.hpp"

namespace {
    Real exponent(const System &system, Real beta) {
        Real g = 0;
        if (system.get_dimensions() == 3) {
            for (const Boson &boson : system.get_bosons()) {
                g += square(boson[0]) + square(boson[1]) + beta * square(boson[2]);
            }
        } else {
            for (const Boson &boson : system.get_bosons()) {
                g += square(boson);
            }
        }
        return g;
    }
}

Real SimpleGaussian::operator() (const System &system) const {
    return std::exp( - _alpha * exponent(system, _beta) );
}

Real SimpleGaussian::derivative_alpha(const System &system) const {
    Real expo = exponent(system, _beta);
    return - std::exp( - _alpha * expo ) * expo;
}

Boson SimpleGaussian::drift_force(const System &system, int boson) const {
    Boson r = system[boson];
    if (system.get_dimensions() == 3) {
        r[2] *= _beta;
    }
    return r * (-4 * _alpha);
}
