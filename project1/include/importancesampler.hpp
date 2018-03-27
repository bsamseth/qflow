#pragma once

#include "definitions.hpp"
#include "system.hpp"
#include "wavefunction.hpp"
#include "sampler.hpp"

/**
 * Class implementing the Metropolis-Hastings algorithm.
 */
class ImportanceSampler : public Sampler {
    protected:
        Boson _q_force_old;
        Boson _q_force_new;

    public:

        /**
         * @copydoc Sampler::Sampler
         */
        ImportanceSampler(const System&, const Wavefunction&, Real step = 0.1);

        virtual void initialize_system();
        virtual void perturb_system();
        virtual Real acceptance_probability() const;

        friend std::ostream& operator<<(std::ostream&, const ImportanceSampler&);
};

inline std::ostream& operator<<(std::ostream &strm, const ImportanceSampler &s) {
    return strm << "ImportanceSampler(dt=" << s._step << ")";
}

