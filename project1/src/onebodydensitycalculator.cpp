#include <cmath>
#include <vector>
#include <string>
#include <fstream>

#include "system.hpp"
#include "wavefunction.hpp"
#include "hamiltonian.hpp"
#include "sampler.hpp"
#include "calculator.hpp"
#include "onebodydensitycalculator.hpp"

OneBodyDensityCalculator::OneBodyDensityCalculator(const Wavefunction &wavefunction,
                                                   const Hamiltonian &hamiltonian,
                                                   Sampler& sampler, std::string logfile,
                                                   int n_bins, Real max_radius)
    : Calculator(wavefunction, hamiltonian, sampler, logfile),
      _max_radius(max_radius), _n_bins(n_bins),
      _density_file(logfile.append("_density.bin"), std::ios::out | std::ios::binary)
{
    _r_step = _max_radius / _n_bins;
    _bins = new long[_n_bins]();
}

void OneBodyDensityCalculator::process_state(const System &system) {
    for (const Boson &boson : system.get_bosons()) {
        Real r_k = std::sqrt(square(boson));
        _bins[std::min(_n_bins - 1, (int) (r_k / _r_step))]++;
    }
}

void OneBodyDensityCalculator::finalize_calculation() {
    for (int i = 0; i < _n_bins; ++i) {
        _density_file.write(reinterpret_cast<const char*>(&_bins[i]), sizeof(_bins[i]));
    }
    _density_file.close();
    delete _bins;
}

