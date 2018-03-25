#pragma once

#include <string>

#include "system.hpp"
#include "wavefunction.hpp"
#include "hamiltonian.hpp"
#include "sampler.hpp"

class EnergyCalculator : public Calculator {

    protected:

        std::ofstream _energy_file;
        bool _analytic;

    public:

        EnergyCalculator(const Wavefunction&, const Hamiltonian&, Sampler&, std::string, bool);
        virtual void process_state(const System&);
        virtual void finalize_calculation();

};
