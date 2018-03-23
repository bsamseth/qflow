#include <string>
#include <fstream>

#include "system.hpp"
#include "wavefunction.hpp"
#include "hamiltonian.hpp"
#include "sampler.hpp"
#include "calculator.hpp"
#include "energycalculator.hpp"

EnergyCalculator::EnergyCalculator(const Wavefunction &wavefunction,
                                   const Hamiltonian &hamiltonian,
                                   Sampler& sampler, std::string logfile, bool analytic)
    : Calculator(wavefunction, hamiltonian, sampler, logfile), _energy_file(logfile.append("_energy.bin"), std::ios::out | std::ios::binary),
      _analytic(analytic)
{
}

void EnergyCalculator::process_state(const System &system) {
    Real E_L;
    if (_analytic) {
        E_L = _hamiltonian.local_energy(system, _wavefunction);
    } else {
        E_L = _hamiltonian.local_energy_numeric(system, _wavefunction);
    }
    _energy_file.write(reinterpret_cast<const char*>(&E_L), sizeof(E_L));
}

void EnergyCalculator::finalize_calculation() {
    _energy_file.close();
}

