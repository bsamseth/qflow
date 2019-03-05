#pragma once

#include "wavefunction.hpp"

#include <initializer_list>
#include <memory>

/**
 * Abstract class for representing a Wavefunction whose parameters are fixed,
 * i.e. will not be updated during an optimization run.
 *
 * All methods simply forward to the underlying wavefunction, with the
 * exception of gradient and set_parameters.
 */
class FixedWavefunction : public Wavefunction
{
private:
    Wavefunction* const psi;

public:
    explicit FixedWavefunction(Wavefunction& wavefunction) : psi(&wavefunction) {}

    Real             operator()(const System& system) override;
    RowVector        gradient(const System& system) override;
    Real             drift_force(const System& system, int k, int dim_index) override;
    RowVector        drift_force(const System& system);
    Real             laplacian(const System& system) override;
    const RowVector& get_parameters() const override;
    void             set_parameters(const RowVector& parameters) override;
    void             set_parameters(std::initializer_list<Real> parameters);
};

inline Real FixedWavefunction::operator()(const System& system)
{
    return psi->operator()(system);
}
inline Real FixedWavefunction::drift_force(const System& system, int k, int dim_index)
{
    return psi->drift_force(system, k, dim_index);
}
inline RowVector FixedWavefunction::drift_force(const System& system)
{
    return psi->drift_force(system);
}
inline Real FixedWavefunction::laplacian(const System& system)
{
    return psi->laplacian(system);
}
inline const RowVector& FixedWavefunction::get_parameters() const
{
    return psi->get_parameters();
}
inline RowVector FixedWavefunction::gradient(const System& system)
{
    SUPPRESS_WARNING(system);
    return RowVector::Zero(psi->get_parameters().size());
}
inline void FixedWavefunction::set_parameters(const RowVector& parameters)
{
    SUPPRESS_WARNING(parameters);
}
inline void FixedWavefunction::set_parameters(std::initializer_list<Real> parameters)
{
    SUPPRESS_WARNING(parameters);
}
