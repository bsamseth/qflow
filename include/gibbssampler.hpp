#pragma once

#include "definitions.hpp"
#include "rbmwavefunction.hpp"
#include "sampler.hpp"
#include "system.hpp"

/**
 * Class implementing the Gibbs sampling algorithm.
 */
class GibbsSampler : public Sampler
{
protected:
    RBMWavefunction& _rbm;
    System           _system;
    Real             _stddev;

    /**
     * @param j index of hidden nodes to compute probability for.
     * @param system configuration to evaluate for.
     * @return P(h_j = 1 | x)
     */
    Real P_h(int j, const System& system) const;

public:
    /**
     * @copydoc Sampler::Sampler
     */
    GibbsSampler(const System&, RBMWavefunction&);

    /**
     * @return A new System instance, drawn from the wavefunction prob. distribution.
     */
    virtual System& next_configuration();

    virtual void initialize_system();

    // These methods are not used in Gibbs, but need to be specified nonetheless.
    virtual void perturb_system();
    virtual Real acceptance_probability() const;
};

inline Real GibbsSampler::P_h(int j, const System& system) const
{
    return 1.0 / (1 + std::exp(-_rbm.v_j(j, system)));
}
inline void GibbsSampler::perturb_system() {};
inline Real GibbsSampler::acceptance_probability() const
{
    return 1;
}
