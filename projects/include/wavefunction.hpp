#pragma once

#include "definitions.hpp"
#include "vector.hpp"
#include "system.hpp"

/**
 * Abstract base class for representing a Wavefunction.
 */
class Wavefunction {

    protected:

        Real _alpha, _beta, _a;

    public:

        /**
         * Initialize wavefunction with the given parameters.
         * @param alpha
         * @param beta
         * @param a Boson hard-sphere radius.
         */
        Wavefunction(Real alpha = 0.5, Real beta = 1, Real a = 0);
        /**
         * Evaluate the wavefunction.
         * @param system System to evaluate the Wavefunction for.
         * @return Value of the wavefunction for the given system.
         */
        virtual Real operator() (System &system) const = 0;
        /**
         * Evaluate d(psi)/d(alpha).
         * @param system System to evaluate for.
         * @return Gradient of wavefunction wrt. alpha.
         */
        virtual Real derivative_alpha(const System &system) const = 0;
        /**
         * Evaluate component of the drift force on a given boson.
         * @param boson Boson to evaluate for.
         * @param dimension Component to evaluate.
         * @return Given component of the drift force on the Boson.
         */
        virtual Real drift_force(const Vector &boson, int dimension) const = 0;
        Real get_alpha() const;
        Real get_beta() const;
        Real get_a() const;
        void set_params(Real alpha, Real beta = 1, Real a = 0);
        virtual ~Wavefunction() = default;
        virtual Real laplacian(System&) const;

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
inline Real Wavefunction::laplacian(System&) const {
    throw std::logic_error("Function not implemented for VMC code.");
}
