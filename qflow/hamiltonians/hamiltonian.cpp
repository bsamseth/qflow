#include "hamiltonian.hpp"

#include "definitions.hpp"
#include "distance.hpp"
#include "mpi.h"
#include "mpiutil.hpp"
#include "sampler.hpp"
#include "system.hpp"

#include <iostream>
#include <limits>

Hamiltonian::Hamiltonian(Real h, Real kinetic_scale_factor)
    : h_(h), kinetic_scale_factor_(kinetic_scale_factor)
{
}

Real Hamiltonian::kinetic_energy_numeric(const System& system, Wavefunction& psi) const
{
    Real E_k = -2 * (system.cols() * system.rows()) * psi(system);

    System& s = const_cast<System&>(system);
    for (int i = 0; i < s.rows(); ++i)
    {
        for (int d = 0; d < s.cols(); ++d)
        {
            const auto temp = s(i, d);
            s(i, d)         = temp + h_;
            Distance::invalidate_cache(i);
            E_k += psi(s);
            s(i, d) = temp - h_;
            Distance::invalidate_cache(i);
            E_k += psi(s);
            s(i, d) = temp;
        }
        Distance::invalidate_cache(i);
    }

    return -0.5 * kinetic_scale_factor_ * E_k / (h_ * h_);
}

Real Hamiltonian::kinetic_energy(const System& system, Wavefunction& psi) const
{
    return -0.5 * kinetic_scale_factor_ * psi.laplacian(system);
}

Real Hamiltonian::local_energy_numeric(const System& system, Wavefunction& psi) const
{
    Real wavefunc = psi(system);
    if (wavefunc == 0)
    {
        return std::numeric_limits<Real>::max();
    }

    return kinetic_energy_numeric(system, psi) / wavefunc + external_potential(system)
           + internal_potential(system);
}

Real Hamiltonian::local_energy(const System& system, Wavefunction& psi) const
{
    return external_potential(system) + internal_potential(system)
           + kinetic_energy(system, psi);
}

Real Hamiltonian::local_energy(Sampler& sampler, Wavefunction& psi, long samples) const
{
    const int  n_procs = mpiutil::proc_count();
    const int  rank    = mpiutil::get_rank();
    const long samples_per_proc
        = samples / n_procs + (rank < samples % n_procs ? 1 : 0);

    Real E_L = 0;

    for (long i = 0; i < samples_per_proc; ++i)
        E_L += local_energy(sampler.next_configuration(), psi);

    Real global_E_L;
    MPI_Allreduce(
        &E_L, &global_E_L, 1, mpiutil::MPI_REAL_TYPE, MPI_SUM, MPI_COMM_WORLD);

    return global_E_L / samples;
}

RowVector Hamiltonian::local_energy_array(Sampler&      sampler,
                                          Wavefunction& psi,
                                          long          samples) const
{
    return generic_array_computation(
        sampler, samples, [&](const System& s) { return local_energy(s, psi); });
}

RowVector Hamiltonian::local_energy_gradient(Sampler&      sampler,
                                             Wavefunction& psi,
                                             long          samples) const
{
    const int  n_procs = mpiutil::proc_count();
    const int  rank    = mpiutil::get_rank();
    const long samples_per_proc
        = samples / n_procs + (rank < samples % n_procs ? 1 : 0);

    Real      E_mean = 0;
    RowVector grad   = RowVector::Zero(psi.get_parameters().size());
    RowVector grad_E = RowVector::Zero(grad.size());
    for (int sample = 0; sample < samples_per_proc; ++sample)
    {
        System& system = sampler.next_configuration();
        Real    E      = local_energy(system, psi);
        E_mean += E;

        RowVector g = psi.gradient(system);
        grad += g;
        grad_E += g * E;
    }

    // Gather results from all workers
    Real      E_mean_global;
    RowVector grad_global(grad.size());
    RowVector grad_E_global(grad_E.size());
    MPI_Allreduce(
        &E_mean, &E_mean_global, 1, mpiutil::MPI_REAL_TYPE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(grad.data(),
                  grad_global.data(),
                  grad.size(),
                  mpiutil::MPI_REAL_TYPE,
                  MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(grad_E.data(),
                  grad_E_global.data(),
                  grad_E.size(),
                  mpiutil::MPI_REAL_TYPE,
                  MPI_SUM,
                  MPI_COMM_WORLD);

    return 2
           * (grad_E_global / samples
              - (grad_global * E_mean_global) / samples / samples);
}

void Hamiltonian::optimize_wavefunction(Wavefunction& psi,
                                        Sampler&      sampler,
                                        int           iterations,
                                        int           sample_points,
                                        SgdOptimizer& optimizer,
                                        Real          gamma,
                                        bool          verbose) const
{
    for (int iteration = 0; iteration < iterations; ++iteration)
    {
        // Thermalize the sampler to the new parameters.
        sampler.thermalize(0.2 * sample_points);

        RowVector grad = local_energy_gradient(sampler, psi, sample_points);

        // Do updates on rank 0
        if (mpiutil::get_rank() == 0)
        {
            if (gamma > 0)
            {
                grad += gamma * psi.get_parameters();
            }
            psi.set_parameters(psi.get_parameters() + optimizer.update_term(grad));
        }

        // Broadcast updated psi to all.
        RowVector params = psi.get_parameters();  // Copy.
        MPI_Bcast(
            params.data(), params.size(), mpiutil::MPI_REAL_TYPE, 0, MPI_COMM_WORLD);
        psi.set_parameters(params);

        if (verbose)
        {
            Real E_mean = local_energy(sampler, psi, sample_points);
            if (mpiutil::get_rank() == 0)
                std::cout << "Iteration " << iteration << ": <E> = " << E_mean << '\n';
        }
    }
}

RowVector Hamiltonian::generic_array_computation(
    Sampler&                             sampler,
    long                                 samples,
    std::function<Real(const System&)>&& func) const
{
    const int n_procs = mpiutil::proc_count();
    const int rank    = mpiutil::get_rank();

    std::vector<int> counts(n_procs, 0);
    std::vector<int> disps(n_procs, 0);
    for (int i = 0; i < n_procs; ++i)
    {
        const long samples_per_proc
            = samples / n_procs + (i < samples % n_procs ? 1 : 0);
        counts[i] = samples_per_proc;
        disps[i]  = i == 0 ? 0 : disps[i - 1] + counts[i - 1];
    }

    RowVector X(counts[rank]);
    RowVector X_all(samples);

    for (long i = 0; i < counts[rank]; ++i)
        X[i] = func(sampler.next_configuration());

    MPI_Allgatherv(X.data(),
                   counts[rank],
                   mpiutil::MPI_REAL_TYPE,
                   X_all.data(),
                   counts.data(),
                   disps.data(),
                   mpiutil::MPI_REAL_TYPE,
                   MPI_COMM_WORLD);

    return X_all;
}

RowVector Hamiltonian::mean_distance_array(Sampler& sampler, long samples) const
{
    assert(sampler.get_current_system().rows() > 1);
    return generic_array_computation(
        sampler, samples, [&](const System& s) { return Distance::probe(s, 0, 1); });
}

RowVector Hamiltonian::mean_radius_array(Sampler& sampler, long samples) const
{
    return generic_array_computation(sampler, samples, [&](const System& s) {
        sampler.thermalize(s.rows());
        Real res = 0;
        for (int j = 0; j < s.rows(); ++j)
            res += norm(s.row(j));
        return res / s.rows();
    });
}

RowVector Hamiltonian::mean_squared_radius_array(Sampler& sampler, long samples) const
{
    return generic_array_computation(sampler, samples, [&](const System& s) {
        sampler.thermalize(s.rows());
        return s.squaredNorm() / s.rows();
    });
}

namespace
{
Real n_dim_volume(Real r_i, Real r_ip1, int dim)
{
    if (dim == 3)
    {
        return (4.0 * PI / 3.0) * (r_ip1 * square(r_ip1) - r_i * square(r_i));
    }
    else if (dim == 2)
    {
        return PI * (square(r_ip1) - square(r_i));
    }
    else
    {
        return r_ip1 - r_i;
    }
}

}  // namespace

RowVector Hamiltonian::onebodydensity(Sampler& sampler,
                                      int      n_bins,
                                      Real     max_radius,
                                      long     samples) const
{
    const int  n_procs = mpiutil::proc_count();
    const int  rank    = mpiutil::get_rank();
    const long samples_per_proc
        = samples / n_procs + (rank < samples % n_procs ? 1 : 0);
    const Real r_step      = max_radius / n_bins;
    RowVector  bins        = RowVector::Zero(n_bins);
    long       total_count = 0;
    const int  n_particles = sampler.get_current_system().rows();

    for (long i = 0; i < samples_per_proc; ++i)
    {
        sampler.thermalize(n_particles);
        const System& system = sampler.get_current_system();
        for (int p = 0; p < n_particles; ++p)
        {
            Real r_k = norm(system.row(p));
            if (r_k < max_radius)
            {
                bins[(int) (r_k / r_step)]++;
                total_count++;
            }
        }
    }

    // Gather results.
    long      global_total_count;
    RowVector global_bins(bins.size());
    MPI_Allreduce(
        &total_count, &global_total_count, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(bins.data(),
                  global_bins.data(),
                  bins.size(),
                  mpiutil::MPI_REAL_TYPE,
                  MPI_SUM,
                  MPI_COMM_WORLD);

    // Normalize counts to physical size of each bin.
    int dimensions = sampler.get_current_system().cols();
    for (int bin = 0; bin < n_bins; ++bin)
    {
        Real r_i   = r_step * bin;
        Real r_ip1 = r_step * (bin + 1);
        global_bins[bin] /= n_dim_volume(r_i, r_ip1, dimensions) * global_total_count;
    }
    return global_bins;
}

Array Hamiltonian::twobodydensity(Sampler& sampler,
                                  int      n_bins,
                                  Real     max_radius,
                                  long     samples) const
{
    const int  n_procs = mpiutil::proc_count();
    const int  rank    = mpiutil::get_rank();
    const long samples_per_proc
        = samples / n_procs + (rank < samples % n_procs ? 1 : 0);
    const Real r_step      = max_radius / n_bins;
    Array      bins        = Array::Zero(n_bins, n_bins);
    long       total_count = 0;
    const int  n_particles = sampler.get_current_system().rows();

    for (long i = 0; i < samples_per_proc; ++i)
    {
        sampler.thermalize(n_particles);
        const System& system = sampler.get_current_system();

        for (int i = 0; i < n_particles - 1; ++i)
        {
            const Real r_i = norm(system.row(i));
            if (r_i >= max_radius)
                continue;

            for (int j = i + 1; j < n_particles; ++j)
            {
                const Real r_j = norm(system.row(j));

                if (r_j < max_radius)
                {
                    bins((int) (r_i / r_step), (int) (r_j / r_step))++;
                    bins((int) (r_j / r_step), (int) (r_i / r_step))++;
                    total_count++;
                }
            }
        }
    }

    // Gather results.
    long  global_total_count;
    Array global_bins(n_bins, n_bins);
    MPI_Allreduce(
        &total_count, &global_total_count, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(bins.data(),
                  global_bins.data(),
                  bins.size(),
                  mpiutil::MPI_REAL_TYPE,
                  MPI_SUM,
                  MPI_COMM_WORLD);

    // Normalize counts to physical size of each bin.
    int dimensions = sampler.get_current_system().cols();
    for (int i = 0; i < n_bins; ++i)
    {
        const Real bin_volume_i
            = n_dim_volume(r_step * i, r_step * (i + 1), dimensions);
        for (int j = 0; j < n_bins; ++j)
        {
            const Real bin_volume_j
                = n_dim_volume(r_step * j, r_step * (j + 1), dimensions);
            global_bins(i, j) /= bin_volume_i * bin_volume_j * global_total_count;
        }
    }
    return global_bins;
}
