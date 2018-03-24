#pragma once

#include <string>

#include "system.hpp"
#include "wavefunction.hpp"
#include "hamiltonian.hpp"
#include "sampler.hpp"

class Calculator {

    protected:

        const Wavefunction &_wavefunction;
        const Hamiltonian &_hamiltonian;
        Sampler &_sampler;
        const std::string _logfile_name;

        virtual void process_state(const System&) = 0;
        virtual void finalize_calculation() = 0;

    public:

        Calculator(const Wavefunction&, const Hamiltonian&, Sampler&, std::string logfile);

        virtual void calculate(long iterations);
};
