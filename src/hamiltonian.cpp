#include <iostream>
#include <limits>

#include "definitions.hpp"
#include "system.hpp"
#include "hamiltonian.hpp"
#include "sampler.hpp"

Hamiltonian::Hamiltonian(Real omega_z, Real a, Real h) : _omega_z(omega_z), _a(a), _h(h) {}

Real Hamiltonian::kinetic_energy_numeric(System &system, const Wavefunction &psi) const {
    Real E_k = -2 * (system.cols() * system.rows()) * psi(system);

    for (int i = 0; i < system.rows(); ++i) {
        for (int d = 0; d < system.cols(); ++d) {
            const auto temp = system(i, d);
            system(i, d) = temp + _h;
            E_k += psi(system);
            system(i, d) = temp - _h;
            E_k += psi(system);
            system(i, d) = temp;
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

RowVector Hamiltonian::local_energy_gradient(Sampler &sampler, const Wavefunction &psi, long samples) const {
    Real E_mean = 0;
    RowVector grad = RowVector::Zero(psi.get_parameters().size());
    RowVector grad_E = RowVector::Zero(grad.size());
    for (int sample = 0; sample < samples; ++sample) {
        System &system = sampler.next_configuration();
        Real E = local_energy(system, psi);
        E_mean += E;

        RowVector g = psi.gradient(system);
        grad += g;
        grad_E += g * E;
    }
    E_mean /= samples;
    grad /= samples;
    grad_E /= samples;

    return 2 * (grad_E - grad * E_mean);
}

void Hamiltonian::optimize_wavefunction(Wavefunction &psi, Sampler &sampler, int iterations,
        int sample_points, SgdOptimizer &optimizer, Real gamma, bool verbose)
{
    for (int iteration = 0; iteration < iterations; ++iteration) {

        // Thermalize the sampler to the new parameters.
        for (int run = 0; run < 0.2 * sample_points; ++run) {
            sampler.next_configuration();
        }

        RowVector grad = local_energy_gradient(sampler, psi, sample_points);

        if (gamma > 0) {
            grad += gamma * psi.get_parameters();
        }

        psi.set_parameters(psi.get_parameters() + optimizer.update_term(grad));

        if (verbose) {
            Real E_mean = 0;
            for (long i = 0; i < sample_points; ++i)
                E_mean += local_energy(sampler.next_configuration(), psi);
            printf("Iteration %d: <E> = %g\n", iteration, E_mean / sample_points);
        }
    }
}

Real Hamiltonian::mean_distance(Sampler &sampler, long samples) const {
    if (sampler.get_current_system().rows() < 2)
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

RowVector Hamiltonian::onebodydensity(Sampler &sampler, int n_bins, Real max_radius, long samples) const {
    Real r_step = max_radius / n_bins;
    RowVector bins = RowVector(n_bins);
    long total_count = 0;

    for (long i = 0; i < samples; ++i) {
        System& system = sampler.next_configuration();
        for (int p = 0; p < system.rows(); ++p) {
            Real r_k = norm(system.row(p));
            if (r_k < max_radius) {
                bins[(int) (r_k / r_step)]++;
                total_count++;
            }
        }
    }

    int dimensions = sampler.get_current_system().cols();
    for (int bin = 0; bin < n_bins; ++bin) {
        Real r_i = r_step * bin;
        Real r_ip1 = r_step * (bin+1);
        bins[bin] /= n_dim_volume(r_i, r_ip1, dimensions) * total_count;
    }
    return bins;
}
