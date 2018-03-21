#pragma once

#include "definitions.hpp"
#include "boson.hpp"
#include "system.hpp"

class Wavefunction {

    Real _alpha, _beta, _a;

    public:

    Wavefunction(Real alpha = 0.5, Real beta = 1, Real a = 0);
    virtual Real operator() (const System&) = 0;
    virtual Real derivative_alpha(const System&) = 0;

    friend std::ostream& operator<<(std::ostream &strm, const Wavefunction &psi);
};

inline std::ostream& operator<<(std::ostream &strm, const Wavefunction &psi) {
    return strm << "Wavefunciton(alpha=" << psi._alpha  << ", beta=" << psi._beta << ", a=" << psi._a << ")";
}
