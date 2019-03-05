#pragma once

#include "definitions.hpp"
#include "sampler.hpp"
#include "system.hpp"
#include "wavefunction.hpp"

/**
 * Class implementing the Metropolis algorithm.
 */
class MetropolisSampler : public Sampler
{
public:
    /**
     * @copydoc Sampler::Sampler
     */
    MetropolisSampler(const System&, Wavefunction&, Real step = 1.0);
    void                 initialize_system() override;
    void                 perturb_system() override;
    Real                 acceptance_probability() const override;
    friend std::ostream& operator<<(std::ostream&, const MetropolisSampler&);
};

inline std::ostream& operator<<(std::ostream& strm, const MetropolisSampler& s)
{
    return strm << "MetropolisSampler(step=" << s._step << ")";
}
