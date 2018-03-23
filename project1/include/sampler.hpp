#pragma once

#include "definitions.hpp"
#include "system.hpp"
#include "wavefunction.hpp"
#include "hamiltonian.hpp"

class Sampler {
    protected:
        const Real _step;
        const Wavefunction *_wavefunction;

        System _system_old;
        System _system_new;
        Real _psi_old;
        Real _psi_new;
        long _accepted_steps = 0;
        long _total_steps = 0;
        int _particle_to_move = 0;

        void prepare_for_next_run();

    public:

        Sampler(const System&, const Wavefunction&, Real step);
        virtual void initialize_system() = 0;
        virtual void perturb_system() = 0;
        virtual Real acceptance_probability() = 0;
        virtual const System& next_configuration();

        Real get_acceptance_rate() const;
        const System& get_current_system() const;

};

inline Real Sampler::get_acceptance_rate() const {
    return _accepted_steps / (Real) _total_steps;
}
inline const System& Sampler::get_current_system() const {
    return _system_old;
}
inline void Sampler::prepare_for_next_run() {
    _total_steps++;
    _particle_to_move = (_particle_to_move + 1) % _system_old.get_n_bosons();
}
