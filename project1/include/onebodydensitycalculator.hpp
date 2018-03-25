#pragma once

#include <string>
#include <vector>

#include "system.hpp"
#include "wavefunction.hpp"
#include "hamiltonian.hpp"
#include "sampler.hpp"

class OneBodyDensityCalculator : public Calculator {

    protected:

        Real _r_step;
        Real _max_radius;
        int _n_bins;

        long *_bins;

        std::ofstream _density_file;

    public:

        OneBodyDensityCalculator(const Wavefunction&, const Hamiltonian&, Sampler&, std::string, int n_bins, Real max_radius);
        virtual void process_state(const System&);
        virtual void finalize_calculation();
};
