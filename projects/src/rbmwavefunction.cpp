#include <vector>
#include <algorithm>
#include <cassert>

#include "prettyprint.hpp"
#include "rbmwavefunction.hpp"


RBMWavefunction::RBMWavefunction(int M, int N, Real sigma2)
    : _sigma2(sigma2),
      _M(M),
      _N(N)
{
    _parameters = Vector(M + N + M*N, rnorm_small_func);
}

Real RBMWavefunction::u_i(int i, System &system) const{
    return square(system.degree(i) - _parameters[a(i)]) / (2 * _sigma2);
}

Real RBMWavefunction::v_j(int j, System &system) const {
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

    return visible * hidden;
}

Real RBMWavefunction::deriv_a(int k, System &system) const {
    return (system.degree(k) - _parameters[a(k)]) / _sigma2;
}

Real RBMWavefunction::deriv_b(int k, System &system) const {
    return 1 / (1 + std::exp(-v_j(k, system)));
}

Real RBMWavefunction::deriv_w(int k, int l, System &system) const {
    return 1 / (1 + std::exp(-v_j(l, system))) * system.degree(k) / _sigma2;
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
            u += square(_parameters[w(k, j)]) * exp_v[j] / (1 + exp_v[j]);
            v += _parameters[w(k, j)] / (1 + exp_v[j]);
        }
        u /= square(_sigma2);

        v = square(_parameters[a(k)] - system.degree(k) + v) / square(_sigma2);

        res += -1 / _sigma2 - u + v;
    }

    return res;
}

Vector RBMWavefunction::gradient(System &system) const {
    Vector grad_vec(_M + _N + _M * _N);

    int k = 0;

    for (int i = 0; i < _M; ++i) {
        grad_vec[k++] = deriv_a(i, system);
    }
    for (int j = 0; j < _N; ++j) {
        grad_vec[k++] = deriv_b(j, system);
    }
    for (int i = 0; i < _M; ++i) {
        for (int j = 0; j < _N; ++j) {
            grad_vec[k++] = deriv_w(i, j, system);
        }
    }

    assert(k == grad_vec.size());

    return grad_vec;
}


void RBMWavefunction::train(const Hamiltonian &hamiltonian,
                            Sampler &sampler,
                            int iterations,
                            int sample_points,
                            Real learning_rate,
                            bool verbose) {


    for (int iteration = 0; iteration < iterations; ++iteration) {
        Vector grad (_M + _N + _M * _N);
        Vector grad_E = grad;
        Vector updates = grad;

        // Thermalize the sampler to the new parameters.
        for (int run = 0; run < sample_points; ++run) {
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

        updates = (-learning_rate * 2) * (grad_E - (E_mean * grad));
        _parameters += -learning_rate * 2 * (grad_E - E_mean * grad);

        if (verbose) {
            printf("Iteration %d: <E> = %g\n", iteration, E_mean);
            std::cout << "params = " << _parameters << '\n';
        }
    }
}








