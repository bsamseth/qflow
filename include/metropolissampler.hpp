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

        MetropolisSampler(const System &init, Wavefunction &wavefunction, Real step = 1, std::size_t N = 1);
        void initialize_system(std::size_t i = 0) override;
        void perturb_system(std::size_t i = 0) override;
        Real acceptance_probability(std::size_t i = 0) const override;
        friend std::ostream& operator<<(std::ostream&, const MetropolisSampler&);
};

inline std::ostream& operator<<(std::ostream &strm, const MetropolisSampler &s) {
    return strm << "MetropolisSampler(step=" << s._step << ")";
}

