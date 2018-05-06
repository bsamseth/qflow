#pragma once

#include "definitions.hpp"
#include "system.hpp"
#include "rbmwavefunction.hpp"
#include "sampler.hpp"

/**
 * Class implementing the Gibbs sampling algorithm.
 */
class GibbsSampler : public Sampler {
    protected:
        const RBMWavefunction &_rbm;
        System _system;
        Real _stddev;

        Real P_h(int j, const System &system) const;

    public:

        /**
         * @copydoc Sampler::Sampler
         */
        GibbsSampler(const System&, const RBMWavefunction&);

        /**
         * @return A new System instance.
         */
        virtual System& next_configuration();

        virtual void initialize_system();
        virtual void perturb_system();
        virtual Real acceptance_probability() const;
};

inline Real GibbsSampler::P_h(int j, const System &system) const {
    return 1.0 / (1 + std::exp(-_rbm.v_j(j, system)));
}
inline void GibbsSampler::perturb_system() {
};
inline Real GibbsSampler::acceptance_probability() const {
    return 1;
}
