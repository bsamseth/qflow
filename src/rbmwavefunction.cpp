#include <vector>
#include <algorithm>
#include <cassert>

#include "prettyprint.hpp"
#include "rbmwavefunction.hpp"
#include "optimizer.hpp"


RBMWavefunction::RBMWavefunction(int M, int N, Real sigma2, Real root_factor)
    : _sigma2(sigma2),
    _root_factor(root_factor),
    _M(M),
    _N(N)
{
    _parameters = Vector::Zero(M + N + M * N);
    for (int i = 0; i < _parameters.size(); ++i) {
        _parameters[i] = rnorm_small_func();
    }
}

Real RBMWavefunction::v_j(int j, const System &system) const {
    Real v = 0;
    for (int i = 0; i < _M; ++i) {
        v += system.data()[i] * _parameters[w(i, j)];
    }
    return _parameters[b(j)] + v / _sigma2;
}

Real RBMWavefunction::operator() (System &system) const {

    // Ensure that system is compatible with rbm, i.e. P*D = M.
    assert(system.rows() * system.cols() == _M);

    Real u = 0;
    for (int i = 0; i < _M; ++i) {
        u += square( system.data()[i] - _parameters[a(i)] );
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
    return _root_factor * (system.data()[k] - _parameters[a(k)]) / _sigma2;
}

Real RBMWavefunction::deriv_b(int k, System &system) const {
    return _root_factor / (1 + std::exp(-v_j(k, system)));
}

Real RBMWavefunction::deriv_w(int k, int l, System &system) const {
    return _root_factor / (1 + std::exp(-v_j(l, system))) * system.data()[k] / _sigma2;
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

        Real dlnPsi  = _root_factor * (_parameters[a(k)] - system.data()[k] + u) / _sigma2;
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
    const int k = system.rows() * particle_index + dim_index;
    Real v = 0;
    for (int j = 0; j < _N; ++j) {
        v += _parameters[w(k, j)] / (1 + std::exp(-v_j(j, system)));
    }
    return _root_factor * 2.0 / _sigma2 * (_parameters[a(k)] - system.data()[k] + v);
}

void RBMWavefunction::train(const Hamiltonian &hamiltonian,
        Sampler &sampler,
        int iterations,
        int sample_points,
        Real learning_rate,
        Real gamma,
        bool verbose) {

    SgdOptimizer optimizer(learning_rate);
    train(hamiltonian, sampler, iterations, sample_points, optimizer, gamma, verbose);
}

void RBMWavefunction::train(const Hamiltonian &hamiltonian,
        Sampler &sampler,
        int iterations,
        int sample_points,
        SgdOptimizer &optimizer,
        Real gamma,
        bool verbose) {

    for (int iteration = 0; iteration < iterations; ++iteration) {

        // Thermalize the sampler to the new parameters.
        for (int run = 0; run < 0.2 * sample_points; ++run) {
            sampler.next_configuration();
        }

        Vector grad = hamiltonian.local_energy_gradient(sampler, *this, sample_points);

        if (gamma > 0) {
            grad += gamma * _parameters;
        }

        _parameters += optimizer.update_term(grad);

        if (verbose) {
            Real E_mean = 0;
            for (long i = 0; i < sample_points; ++i)
                E_mean += hamiltonian.local_energy(sampler.next_configuration(), *this);
            printf("Iteration %d: <E> = %g\n", iteration, E_mean / sample_points);
        }
    }
}
