#pragma once
#include <iostream>

#include "definitions.hpp"
#include "system.hpp"
#include "wavefunction.hpp"

/**
 * Abstract sampler class defining the iterface for a generic Monte Carlo sampler.
 */
class Sampler {
    protected:
        const Real _step;
        Wavefunction *_wavefunction;
        const std::size_t _N_instances;

        struct StateInfo {
            System system_old;
            System system_new;
            Real psi_old;
            Real psi_new;
            long accepted_steps = 0;
            long total_steps = 0;
            int particle_to_move = 0;

            StateInfo(System s, Wavefunction &wavefunction);
        };

        std::vector<StateInfo> _instances;

        void prepare_for_next_run(std::size_t i);

    public:

        /**
         * Initialize a sampler.
         * @param init Shape of System instances to generate.
         * @param wavefunction Wavefunction to sample from.
         * @param step Step size to use in sampling.
         * @param N Number of simultaneous states to set up.
         */
        Sampler(const System &init, Wavefunction &wavefunction, Real step = 1, std::size_t N = 1);

        /**
         * Initialize system in some random configuration.
         */
        virtual void initialize_system(std::size_t i = 0) = 0;
        /**
         * Perturb the system, producing a new suggested state.
         */
        virtual void perturb_system(std::size_t i = 0) = 0;
        /**
         * Compute the acceptance probability for the newly generated state.
         */
        virtual Real acceptance_probability(std::size_t i = 0) const = 0;
        /**
         * @return A new System instance.
         */
        virtual System& next_configuration(std::size_t i = 0);

        /**
         * @return Number of accepted steps.
         */
        long get_accepted_steps(std::size_t i = 0) const;
        /**
         * @return Total number of calls to next_configuration.
         */
        long get_total_steps(std::size_t i = 0) const;
        /**
         * @return Acceptance rate.
         */
        Real get_acceptance_rate(std::size_t i = 0) const;
        /**
         * @return The current system of the sampler.
         */
        const System& get_current_system(std::size_t i = 0) const;

        friend std::ostream& operator<<(std::ostream &strm, const Sampler& s);
};

inline Real Sampler::get_acceptance_rate(std::size_t i) const {
    return _instances[i].accepted_steps / (Real) _instances[i].total_steps;
}
inline long Sampler::get_accepted_steps(std::size_t i) const {
    return _instances[i].accepted_steps;
}
inline long Sampler::get_total_steps(std::size_t i) const {
    return _instances[i].total_steps;
}
inline const System& Sampler::get_current_system(std::size_t i) const {
    return _instances[i].system_old;
}
inline void Sampler::prepare_for_next_run(std::size_t i) {
    _instances[i].total_steps++;
    _instances[i].particle_to_move = (_instances[i].particle_to_move + 1) % _instances[i].system_old.rows();
}
inline std::ostream& operator<<(std::ostream &strm, const Sampler& s) {
    return strm << "Sampler(step=" << s._step << ")";
}
