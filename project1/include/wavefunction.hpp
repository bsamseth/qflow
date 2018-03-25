#pragma once

#include "definitions.hpp"
#include "boson.hpp"
#include "system.hpp"

class Wavefunction {

    protected:

        Real _alpha, _beta, _a;

    public:

        Wavefunction(Real alpha = 0.5, Real beta = 1, Real a = 0);
        virtual Real operator() (const System&) const = 0;
        virtual Real derivative_alpha(const System&) const = 0;
        virtual Real drift_force(const Boson&, int) const = 0;
        Real get_alpha() const;
        Real get_beta() const;
        Real get_a() const;
        void set_params(Real alpha, Real beta = 1, Real a = 0);

        friend std::ostream& operator<<(std::ostream &strm, const Wavefunction &psi);
};

inline Real Wavefunction::get_alpha() const {
    return _alpha;
}
inline Real Wavefunction::get_beta() const {
    return _beta;
}
inline Real Wavefunction::get_a() const {
    return _a;
}
inline void Wavefunction::set_params(Real alpha, Real beta, Real a) {
    _alpha = alpha; _beta = beta; _a = a;
}
inline std::ostream& operator<<(std::ostream &strm, const Wavefunction &psi) {
    return strm << "Wavefunciton(alpha=" << psi._alpha  << ", beta=" << psi._beta << ", a=" << psi._a << ")";
}
