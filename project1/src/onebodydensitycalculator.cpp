#include <cmath>
#include <vector>
#include <string>
#include <fstream>

#include "definitions.hpp"
#include "system.hpp"
#include "wavefunction.hpp"
#include "hamiltonian.hpp"
#include "sampler.hpp"
#include "calculator.hpp"
#include "onebodydensitycalculator.hpp"

namespace {
    Real n_dim_volume(Real r_i, Real r_ip1, int dim) {
        if (dim == 3) {
            return (4.0 * PI / 3.0) * (r_ip1 * square(r_ip1) - r_i * square(r_i));
        } else if (dim == 2) {
            return PI * (square(r_ip1) - square(r_i));
        } else {
            return r_ip1 - r_i;
        }
    }
}


OneBodyDensityCalculator::OneBodyDensityCalculator(const Wavefunction &wavefunction,
                                                   const Hamiltonian &hamiltonian,
                                                   Sampler& sampler, std::string logfile,
                                                   int n_bins, Real max_radius)
    : Calculator(wavefunction, hamiltonian, sampler, logfile),
      _max_radius(max_radius), _n_bins(n_bins),
      _density_file(logfile.append("_density.bin"), std::ios::out | std::ios::binary)
{
    _r_step = _max_radius / _n_bins;
    _bins = new Real[_n_bins]();
}

void OneBodyDensityCalculator::process_state(const System &system) {
    for (const Boson &boson : system.get_bosons()) {
        Real r_k = std::sqrt(square(boson));
        if (r_k < _max_radius) {
            _bins[(int) (r_k / _r_step)]++;
            _total_count++;
        }
    }
}

void OneBodyDensityCalculator::finalize_calculation() {
    // Normalize counts to volume.
    for (int bin = 0; bin < _n_bins; ++bin) {
        Real r_i = _r_step * bin;
        Real r_ip1 = _r_step * (bin+1);
        _bins[bin] /= n_dim_volume(r_i, r_ip1, _sampler.get_current_system().get_dimensions());
        _bins[bin] /= _total_count;
    }

    for (int i = 0; i < _n_bins; ++i) {
        _density_file.write(reinterpret_cast<const char*>(&_bins[i]), sizeof(_bins[i]));
    }
    _density_file.close();
    delete _bins;
}

