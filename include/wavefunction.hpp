#pragma once

#include "definitions.hpp"
#include "system.hpp"
#include "vector.hpp"

#include <initializer_list>

class Sampler;

/**
 * Abstract base class for representing a Wavefunction.
 */
class Wavefunction
{
protected:
    RowVector _parameters;

public:
    Wavefunction() = default;
    Wavefunction(const RowVector& parameters);

    /**
     * Evaluate the wavefunction.
     * @param system System to evaluate the Wavefunction for.
     * @return Value of the wavefunction for the given system.
     */
    virtual Real operator()(const System& system) = 0;

    /**
     * Return grad(Psi)/Psi, with derivatives wrt. all parameters.
     * @param system System configuration to evaluate gradient for.
     * @return RowVector with derivatives.
     */
    virtual RowVector gradient(const System& system) = 0;

    /**
     * Return 2/Psi * d(Psi)/dx_{k, dim_index}.
     */
    virtual Real drift_force(const System& system, int k, int dim_index);

    RowVector drift_force(const System& system);

    /**
     * Return sum_{particles} laplacian(Psi)/Psi.
     * @param system System configuration to evaluate laplacian for.
     * @return The sum of laplacians for all particles in the system.
     */
    virtual Real laplacian(const System& system) = 0;

    Real symmetry_metric(Sampler& sampler, long samples, int max_permutations = 100);

    virtual const RowVector& get_parameters() const;

    virtual void set_parameters(const RowVector& parameters);

    void set_parameters(std::initializer_list<Real> parameters);

    virtual ~Wavefunction() = default;

    friend std::ostream& operator<<(std::ostream& strm, const Wavefunction& psi);
};

inline const RowVector& Wavefunction::get_parameters() const
{
    return _parameters;
}
inline void Wavefunction::set_parameters(const RowVector& parameters)
{
    _parameters = parameters;
}
inline void Wavefunction::set_parameters(std::initializer_list<Real> parameters)
{
    set_parameters(vector_from_sequence(parameters));
}
inline Real Wavefunction::drift_force(const System& system, int k, int dim_index)
{
    SUPPRESS_WARNING(system);
    SUPPRESS_WARNING(k);
    SUPPRESS_WARNING(dim_index;)
    throw std::logic_error("Drift force by default not defined.");
}
inline std::ostream& operator<<(std::ostream& strm, const Wavefunction& psi)
{
    return strm << "Wavefunction(" << psi._parameters << ")";
}
