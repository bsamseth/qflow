#include <vector>
#include <algorithm>
#include <cassert>

#include "prettyprint.hpp"
#include "rbmwavefunction.hpp"


RBMWavefunction::RBMWavefunction(int M, int N, Real sigma2, Real root_factor)
    : _sigma2(sigma2),
    _root_factor(root_factor),
    _M(M),
    _N(N)
{
    _parameters = Vector(M + N + M*N, rnorm_small_func);
}

Real RBMWavefunction::v_j(int j, const System &system) const {
    Real v = 0;
    for (int i = 0; i < _M; ++i) {
        v += system.degree(i) * _parameters[w(i, j)];
    }
    return _parameters[b(j)] + v / _sigma2;
}

Real RBMWavefunction::operator() (System &system) const {

    // Ensure that system is compatible with rbm, i.e. P*D = M.
    assert(system.get_dimensions() * system.get_n_particles() == _M);

    Real u = 0;
    for (int i = 0; i < _M; ++i) {
        u += square( system.degree(i) - _parameters[a(i)] );
    }
    Real visible = std::exp(- 0.5 * u / _sigma2);

    Real hidden = 1;
    for (int j = 0; j < _N; ++j) {
        hidden *= (1 + std::exp(v_j(j, system)));
    }

    if (_root_factor < 1)
        return std::sqrt(visible * hidden);
    else
        return visible * hidden;
}

Real RBMWavefunction::deriv_a(int k, System &system) const {
    return _root_factor * (system.degree(k) - _parameters[a(k)]) / _sigma2;
}

Real RBMWavefunction::deriv_b(int k, System &system) const {
    return _root_factor / (1 + std::exp(-v_j(k, system)));
}

Real RBMWavefunction::deriv_w(int k, int l, System &system) const {
    return _root_factor / (1 + std::exp(-v_j(l, system))) * system.degree(k) / _sigma2;
}

Real RBMWavefunction::laplacian(System &system) const {

    std::vector<Real> exp_v(_N);
    for (int j = 0; j < _N; ++j) {
        exp_v[j] = std::exp(-v_j(j, system));
    }

    Real res = 0;
    for (int k = 0; k < _M; ++k) {

        Real u = 0, v = 0;
        for (int j = 0; j < _N; ++j) {
            u += _parameters[w(k, j)]  / (1 + exp_v[j]);
            v += square(_parameters[w(k, j)]) * exp_v[j] / square(1 + exp_v[j]);
        }

        Real dlnPsi  = _root_factor * (_parameters[a(k)] - system.degree(k) + u) / _sigma2;
        Real ddlnPsi = _root_factor * (- 1. / _sigma2 + v / square(_sigma2));

        res += square(dlnPsi) + ddlnPsi;
    }

    return res;
}

Vector RBMWavefunction::gradient(System &system) const {
    Vector grad_vec(_parameters.size());

    for (int i = 0; i < _M; ++i) {
        grad_vec[a(i)] = deriv_a(i, system);
    }
    for (int j = 0; j < _N; ++j) {
        grad_vec[b(j)] = deriv_b(j, system);
    }
    for (int i = 0; i < _M; ++i) {
        for (int j = 0; j < _N; ++j) {
            grad_vec[w(i, j)] = deriv_w(i, j, system);
        }
    }

    return grad_vec;
}

Real RBMWavefunction::drift_force(const System &system, int particle_index, int dim_index) const {
    const int k = system.get_dimensions() * particle_index + dim_index;
    Real v = 0;
    for (int j = 0; j < _N; ++j) {
        v += _parameters[w(k, j)] / (1 + std::exp(-v_j(j, system)));
    }
    return _root_factor * 2.0 / _sigma2 * (_parameters[a(k)] - system.degree(k) + v);
}


void RBMWavefunction::train(const Hamiltonian &hamiltonian,
        Sampler &sampler,
        int iterations,
        int sample_points,
        Real learning_rate,
        Real gamma,
        bool verbose) {


    for (int iteration = 0; iteration < iterations; ++iteration) {
        Vector grad (_M + _N + _M * _N);
        Vector grad_E = grad;

        // Thermalize the sampler to the new parameters.
        for (int run = 0; run < 0.2 * sample_points; ++run) {
            sampler.next_configuration();
        }

        Real E_mean = 0;

        for (int sample = 0; sample < sample_points; ++sample) {
            System &system = sampler.next_configuration();
            Real E = hamiltonian.local_energy(system, *this);
            E_mean += E;

            Vector g = gradient(system);
            grad += g;
            grad_E += g * E;
        }

        E_mean /= sample_points;
        grad /= sample_points;
        grad_E /= sample_points;

        grad *= E_mean;
        grad_E -= grad;
        grad_E *= - learning_rate * 2;

        _parameters += grad_E;

        if (gamma > 0) {
            _parameters -= gamma * 2 * _parameters;
        }

        if (verbose) {
            printf("Iteration %d: <E> = %g\n", iteration, E_mean);
            std::cout << _parameters << std::endl;
            //std::cout << "params = " << _parameters << '\n';
        }
    }
}

