#pragma once

#include <string>
#include <vector>

#include "system.hpp"
#include "wavefunction.hpp"
#include "hamiltonian.hpp"
#include "sampler.hpp"
#include "calculator.hpp"

/**
 * Calculator for computing one-body densities.
 */
class OneBodyDensityCalculator : public Calculator {

    protected:

        Real _r_step;
        Real _max_radius;
        int _n_bins;
        long _total_count = 0;

        Real *_bins;

        std::ofstream _density_file;

    public:

        /**
         * @copydoc Calculator::Calculator
         * @param n_bins Number of regions to split space into.
         * @param max_radius Maximum radius for regions.
         */
        OneBodyDensityCalculator(const Wavefunction&, const Hamiltonian&, Sampler&, std::string, int n_bins, Real max_radius);
        virtual void process_state(System&);
        virtual void finalize_calculation();
};
