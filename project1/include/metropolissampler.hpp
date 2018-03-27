#pragma once

#include "definitions.hpp"
#include "system.hpp"
#include "wavefunction.hpp"
#include "sampler.hpp"

/**
 * Class implementing the Metropolis algorithm.
 */
class MetropolisSampler : public Sampler {
    public:

        /**
         * @copydoc Sampler::Sampler
         */
        MetropolisSampler(const System&, const Wavefunction&, Real step = 1.0);
        virtual void initialize_system();
        virtual void perturb_system();
        virtual Real acceptance_probability() const;
        friend std::ostream& operator<<(std::ostream&, const MetropolisSampler&);
};

inline std::ostream& operator<<(std::ostream &strm, const MetropolisSampler &s) {
    return strm << "MetropolisSampler(step=" << s._step << ")";
}

