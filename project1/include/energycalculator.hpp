#pragma once

#include <string>

#include "system.hpp"
#include "wavefunction.hpp"
#include "hamiltonian.hpp"
#include "sampler.hpp"

/**
 * Calculator used to calculate local energies.
 */
class EnergyCalculator : public Calculator {

    protected:

        std::ofstream _energy_file;
        bool _analytic;

    public:

        /**
         * @copydoc Calculator::Calculator
         * @param analytic False will use numerical approximation of local energies.
         */
        EnergyCalculator(const Wavefunction&, const Hamiltonian&, Sampler&, std::string, bool analytic);

        virtual void process_state(System &);
        virtual void finalize_calculation();

};
