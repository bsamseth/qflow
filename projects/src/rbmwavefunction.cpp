#include <vector>
#include <cassert>

#include "prettyprint.hpp"
#include "rbmwavefunction.hpp"



RBMWavefunction::RBMWavefunction(int M, int N, Real sigma2)
    : _sigma2(sigma2),
      _M(M),
      _N(N),
      _a(M),
      _b(N),
      _w(M, std::vector<Real>(N))
{
    std::generate(_a.begin(), _a.end(), rnorm_func);
    std::generate(_b.begin(), _b.end(), rnorm_func);
    for (auto &row : _w) {
        std::generate(row.begin(), row.end(), rnorm_func);
    }
}

Real RBMWavefunction::u_i(int i, System &system) const{
    return square(system.degree(i) - _a[i]) / (2 * _sigma2);
}

Real RBMWavefunction::v_j(int j, System &system) const {
    Real v = 0;
    for (int i = 0; i < _M; ++i) {
        v += system.degree(i) * _w[i][j];
    }
    return _b[j] + v / _sigma2;
}

Real RBMWavefunction::operator() (System &system) const {

    assert(system.get_dimensions() * system.get_n_bosons() == _M);

    Real u = 0;
    for (int i = 0; i < _M; ++i) {
        auto x_i = system.degree(i);
        u += square( x_i - _a[i] );
    }
    Real visible = std::exp(- 0.5 * u / _sigma2);

    Real hidden = 1;
    for (int j = 0; j < _N; ++j) {
        hidden *= (1 + std::exp(v_j(j, system)));
    }

    return visible * hidden;
}

Real RBMWavefunction::deriv_a(int k, System &system) const {
    return (system.degree(k) - _a[k]) / _sigma2;
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
            u += square(_w[k][j]) * exp_v[j] / (1 + exp_v[j]);
            v += _w[k][j] / (1 + exp_v[j]);
        }
        u /= square(_sigma2);

        v = square(_a[k] -system.degree(k) + v) / square(_sigma2);

        res += -1 / _sigma2 + u + v;
    }

    return res;
}

void RBMWavefunction::gradient(System &system, std::vector<Real> &grad_vec) const {
    assert((int)grad_vec.size() == _M + _N + _M * _N);

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

    assert(k == (int) grad_vec.size());

}
void RBMWavefunction::update_params(std::vector<Real> &grad_vec) {
    assert((int)grad_vec.size() == _M + _N + _M * _N);

    int k = 0;

    for (int i = 0; i < _M; ++i) {
        _a[i] += grad_vec[k++];
    }
    for (int j = 0; j < _N; ++j) {
        _b[j] += grad_vec[k++];
    }
    for (int i = 0; i < _M; ++i) {
        for (int j = 0; j < _N; ++j) {
            _w[i][j] += grad_vec[k++];
        }
    }

    assert(k == (int) grad_vec.size());
}

void RBMWavefunction::set_params(const std::vector<Real> &param_vec) {
    assert((int)param_vec.size() == _M + _N + _M * _N);

    int k = 0;

    for (int i = 0; i < _M; ++i) {
        _a[i] = param_vec[k++];
    }
    for (int j = 0; j < _N; ++j) {
        _b[j] = param_vec[k++];
    }
    for (int i = 0; i < _M; ++i) {
        for (int j = 0; j < _N; ++j) {
            _w[i][j] = param_vec[k++];
        }
    }

    assert(k == (int) param_vec.size());
}



void RBMWavefunction::train(const Hamiltonian &hamiltonian,
                            Sampler &sampler,
                            int iterations,
                            int sample_points,
                            Real learning_rate) {


    for (int iteration = 0; iteration < iterations; ++iteration) {
        std::vector<Real> grad (_M + _N + _M * _N, 0);
        std::vector<Real> grad_E = grad;
        std::vector<Real> updates = grad;

        // Thermalize the sampler to the new parameters.
        for (int run = 0; run < sample_points; ++run) {
            sampler.next_configuration();
        }

        Real E_mean = 0;

        for (int sample = 0; sample < sample_points; ++sample) {
            System &system = sampler.next_configuration();
            Real E = hamiltonian.local_energy_numeric(system, *this);
            E_mean += E;


            std::size_t k = 0;
            for (int i = 0; i < _M; ++i, ++k) {
                auto d = deriv_a(i, system);
                grad[k] += d;
                grad_E[k] += E * d;
            }
            for (int j = 0; j < _N; ++j, ++k) {
                auto d = deriv_b(j, system);
                grad[k] += d;
                grad_E[k] += E * d;
            }
            for (int i = 0; i < _M; ++i) {
                for (int j = 0; j < _N; ++j, ++k) {
                    auto d = deriv_w(i, j, system);
                    grad[k] += d;
                    grad_E[k] += E * d;
                }
            }

            assert(k == grad.size());
        }

        E_mean /= sample_points;
        for (std::size_t i = 0; i < grad.size(); ++i) {
            grad[i] /= sample_points;
            grad_E[i] /= sample_points;

            updates[i] = - learning_rate * 2 * (grad_E[i] - E_mean * grad[i]);
        }


        update_params(updates);


        printf("Iteration %d: <E> = %g\n", iteration, E_mean);
        std::cout << "updates: " << updates << '\n';
        std::cout << "a = " << _a << "\n";
        std::cout << "b = " << _b << "\n";
        std::cout << "w = " << _w << "\n";


    }
}








