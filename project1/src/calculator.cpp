#include <string>

#include "system.hpp"
#include "wavefunction.hpp"
#include "hamiltonian.hpp"
#include "sampler.hpp"
#include "calculator.hpp"

Calculator::Calculator(const Wavefunction &wavefunction,
                       const Hamiltonian &hamiltonian,
                       Sampler& sampler, std::string logfile)
    : _wavefunction(wavefunction), _hamiltonian(hamiltonian),
      _sampler(sampler), _logfile_name(logfile) { }


void Calculator::calculate(long iterations) {

    for (int iter = 0; iter < iterations; ++iter) {
        process_state(_sampler.next_configuration());
    }

    finalize_calculation();
}
