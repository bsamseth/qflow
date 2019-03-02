#pragma once

#include "definitions.hpp"
#include "sampler.hpp"
#include "system.hpp"
#include "wavefunction.hpp"

/**
 * Class implementing the Metropolis-Hastings algorithm.
 */
class ImportanceSampler : public Sampler
{
public:
    /**
     * @copydoc Sampler::Sampler
     */
    ImportanceSampler(const System&, Wavefunction&, Real step = 0.1);

    void initialize_system() override;
    void perturb_system() override;
    Real acceptance_probability() const override;

    friend std::ostream& operator<<(std::ostream&, const ImportanceSampler&);
};

inline std::ostream& operator<<(std::ostream& strm, const ImportanceSampler& s)
{
    return strm << "ImportanceSampler(dt=" << s._step << ")";
}
