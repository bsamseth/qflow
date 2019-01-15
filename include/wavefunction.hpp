#pragma once

#include <initializer_list>

#include "definitions.hpp"
#include "vector.hpp"
#include "system.hpp"

/**
 * Abstract base class for representing a Wavefunction.
 */
class Wavefunction {

    protected:

        RowVector _parameters;

    public:

        /**
         * Initialize wavefunction with the given parameters.
         * @param parameters Initializer list of parameters.
         */
        Wavefunction(std::initializer_list<Real> parameters = {});

        /**
         * Evaluate the wavefunction.
         * @param system System to evaluate the Wavefunction for.
         * @return Value of the wavefunction for the given system.
         */
        virtual Real operator() (System &system) const = 0;

        /**
         * Return grad(Psi)/Psi, with derivatives wrt. all parameters.
         * @param system System configuration to evaluate gradient for.
         * @return RowVector with derivatives.
         */
        virtual RowVector gradient(System &system) const = 0;


        virtual Real drift_force(const System &system, int k, int dim_index) const;

        /**
         * Return sum_{particles} laplacian(Psi)/Psi.
         * @param system System configuration to evaluate laplacian for.
         * @return The sum of laplacians for all particles in the system.
         */
        virtual Real laplacian(System &system) const = 0;

        RowVector& get_parameters();
        const RowVector& get_parameters() const;

        void set_parameters(const RowVector &parameters);
        void set_parameters(std::initializer_list<Real> parameters);

        virtual ~Wavefunction() = default;

        friend std::ostream& operator<<(std::ostream &strm, const Wavefunction &psi);
};

inline RowVector& Wavefunction::get_parameters() {
    return _parameters;
}
inline const RowVector& Wavefunction::get_parameters() const {
    return _parameters;
}
inline void Wavefunction::set_parameters(const RowVector &parameters) {
    _parameters = parameters;
}
inline void Wavefunction::set_parameters(std::initializer_list<Real> parameters) {
    set_parameters(vector_from_sequence(parameters));
}
inline Real Wavefunction::drift_force(const System &system, int k, int dim_index) const {
    SUPPRESS_WARNING(system);
    SUPPRESS_WARNING(k);
    SUPPRESS_WARNING(dim_index;)
    throw std::logic_error("Drift force by default not defined.");
}
inline std::ostream& operator<<(std::ostream &strm, const Wavefunction &psi) {
    return strm << "Wavefunction(" << psi._parameters << ")";
}
