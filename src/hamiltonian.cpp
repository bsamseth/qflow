#include <iostream>
#include <limits>

#include "definitions.hpp"
#include "system.hpp"
#include "hamiltonian.hpp"
#include "sampler.hpp"

Hamiltonian::Hamiltonian(Real omega_z, Real a, Real h) : _omega_z(omega_z), _a(a), _h(h) {}

Real Hamiltonian::kinetic_energy_numeric(System &system, const Wavefunction &psi) const {
    Real E_k = -2 * (system.cols() * system.rows()) * psi(system);

    for (int i = 0; i < system.cols(); ++i) {
        for (int d = 0; d < system.rows(); ++d) {
            const auto temp = system(d, i);
            system(d, i) = temp + _h;
            E_k += psi(system);
            system(d, i) = temp - _h;
            E_k += psi(system);
            system(d, i) = temp;
        }
    }

    return -0.5 * E_k / (_h * _h);
}

Real Hamiltonian::kinetic_energy(System &system, const Wavefunction &psi) const {
    return -0.5 * psi.laplacian(system);
}

Real Hamiltonian::local_energy_numeric(System &system, const Wavefunction &psi) const {
    Real wavefunc = psi(system);
    if (wavefunc == 0) {
        return std::numeric_limits<Real>::max();
    }

    return kinetic_energy_numeric(system, psi) / wavefunc
         + external_potential(system)
         + internal_potential(system);
}

Real Hamiltonian::local_energy(System &system, const Wavefunction &psi) const {
    return external_potential(system) + internal_potential(system) + kinetic_energy(system, psi);
}

Vector Hamiltonian::local_energy_gradient(Sampler &sampler, const Wavefunction &psi, long samples) const {
    Real E_mean = 0;
    Vector grad = Vector::Zero(psi.get_parameters().size());
    Vector grad_E = Vector::Zero(psi.get_parameters().size());
    for (int sample = 0; sample < samples; ++sample) {
        System &system = sampler.next_configuration();
        Real E = local_energy(system, psi);
        E_mean += E;

        Vector g = psi.gradient(system);
        grad += g;
        grad_E += g * E;
    }
    E_mean /= samples;
    grad /= samples;
    grad_E /= samples;

    grad *= E_mean;
    grad_E -= grad;
    grad_E *= 2;
    return grad_E;
}

Real Hamiltonian::mean_distance(Sampler &sampler, long samples) const {
    if (sampler.get_current_system().cols() < 2)
        return 0;

    Real dist = 0;
    for (long i = 0; i < samples; ++i) {
        dist += distance(sampler.next_configuration(), 0, 1);
    }
    return dist / samples;
}

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

Vector Hamiltonian::onebodydensity(Sampler &sampler, int n_bins, Real max_radius, long samples) const {
    Real r_step = max_radius / n_bins;
    Vector bins = Vector(n_bins);
    long total_count = 0;

    for (long i = 0; i < samples; ++i) {
        System& system = sampler.next_configuration();
        for (int p = 0; p < system.cols(); ++p) {
            Real r_k = norm(system.col(p));
            if (r_k < max_radius) {
                bins[(int) (r_k / r_step)]++;
                total_count++;
            }
        }
    }

    int dimensions = sampler.get_current_system().rows();
    for (int bin = 0; bin < n_bins; ++bin) {
        Real r_i = r_step * bin;
        Real r_ip1 = r_step * (bin+1);
        bins[bin] /= n_dim_volume(r_i, r_ip1, dimensions) * total_count;
    }
    return bins;
}