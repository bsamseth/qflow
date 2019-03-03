#pragma once
#include "definitions.hpp"
#include "system.hpp"
#include "wavefunction.hpp"

#include <iostream>

/**
 * Abstract sampler class defining the iterface for a generic Monte Carlo sampler.
 */
class Sampler
{
protected:
    const Real    _step;
    Wavefunction* _wavefunction;

    System _system_old;
    System _system_new;
    Real   _psi_old;
    Real   _psi_new;
    long   _accepted_steps   = 0;
    long   _total_steps      = 0;
    int    _particle_to_move = 0;

    void prepare_for_next_run();

public:
    /**
     * Initialize a sampler.
     * @param init Shape of System instances to generate.
     * @param wavefunction Wavefunction to sample from.
     * @param step Step size to use in sampling.
     */
    Sampler(const System& init, Wavefunction& wavefunction, Real step);

    /**
     * Initialize system in some random configuration.
     */
    virtual void initialize_system() = 0;
    /**
     * Perturb the system, producing a new suggested state.
     */
    virtual void perturb_system() = 0;
    /**
     * Compute the acceptance probability for the newly generated state.
     */
    virtual Real acceptance_probability() const = 0;
    /**
     * @return A new System instance.
     */
    virtual System& next_configuration();

    /**
     * @return Number of accepted steps.
     */
    long get_accepted_steps() const;
    /**
     * @return Total number of calls to next_configuration.
     */
    long get_total_steps() const;
    /**
     * @return Acceptance rate.
     */
    Real get_acceptance_rate() const;
    /**
     * @return The current system of the sampler.
     */
    const System& get_current_system() const;

    void thermalize(long samples);

    friend std::ostream& operator<<(std::ostream& strm, const Sampler& s);
};

inline Real Sampler::get_acceptance_rate() const
{
    return _accepted_steps / (Real) _total_steps;
}
inline long Sampler::get_accepted_steps() const
{
    return _accepted_steps;
}
inline long Sampler::get_total_steps() const
{
    return _total_steps;
}
inline const System& Sampler::get_current_system() const
{
    return _system_old;
}
inline void Sampler::prepare_for_next_run()
{
    _total_steps++;
    _particle_to_move = (_particle_to_move + 1) % _system_old.rows();
}
inline std::ostream& operator<<(std::ostream& strm, const Sampler& s)
{
    return strm << "Sampler(step=" << s._step << ")";
}
