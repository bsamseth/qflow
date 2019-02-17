#pragma once

#include "definitions.hpp"
#include "system.hpp"
#include "wavefunction.hpp"
#include "sampler.hpp"

/**
 * Class implementing the Metropolis-Hastings algorithm.
 */
class ImportanceSampler : public Sampler {
    public:

        /**
         * @copydoc Sampler::Sampler
         */
        ImportanceSampler(const System&, Wavefunction&, Real step = 0.1, std::size_t N = 1);

        void initialize_system(std::size_t i) override;
        void perturb_system(std::size_t i) override;
        Real acceptance_probability(std::size_t i) const override;

        friend std::ostream& operator<<(std::ostream&, const ImportanceSampler&);
};

inline std::ostream& operator<<(std::ostream &strm, const ImportanceSampler &s) {
    return strm << "ImportanceSampler(dt=" << s._step << ")";
}

