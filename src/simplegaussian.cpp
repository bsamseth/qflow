#include <cmath>

#include "definitions.hpp"
#include "system.hpp"
#include "simplegaussian.hpp"

Real SimpleGaussian::operator() (const System &system) {
    Real g = 0;
    if (system.get_dimensions() == 3) {
        for (const Boson &boson : system.get_bosons()) {
            g += square(boson[0]) + square(boson[1]) + _beta * square(boson[2]);
        }
    } else {
        for (int i = 0; i < system.get_n_bosons(); ++i) {
            g += system[i] * system[i];
        }
    }
    return std::exp( - _alpha * g);
}

Real SimpleGaussian::derivative_alpha(const System &system) {
    return -123;
}
