#pragma once

#include <vector>
#include <algorithm>
#include <functional>
#include <cassert>

#include "definitions.hpp"
#include "system.hpp"
#include "wavefunction.hpp"
#include "sampler.hpp"
#include "hamiltonian.hpp"

class RBMWavefunction : public Wavefunction {

    protected:

        const Real _sigma2;
        const int _M;
        const int _N;

        std::vector<Real> _a;
        std::vector<Real> _b;
        std::vector<std::vector<Real>> _w;

        Real u_i(int i, System &system) const;
        Real v_j(int j, System &system) const;

    public:

        RBMWavefunction(int M, int P, Real sigma = 1);

        void gradient(System &system, std::vector<Real> &grad_vec) const;

        void update_params(std::vector<Real> &grad_vec);

        void set_params(const std::vector<Real> &param_vec);

        virtual Real operator() (System &system) const;

        virtual Real deriv_a(int k, System &system) const;

        virtual Real deriv_b(int k, System &system) const;

        virtual Real deriv_w(int k, int l, System &system) const;

        virtual Real laplacian(System &system) const;

        virtual void train(const Hamiltonian &hamiltonian,
                           Sampler &sampler,
                           int iterations,
                           int sample_points,
                           Real learning_rate);

        virtual Real derivative_alpha(const System&) const;
        virtual Real drift_force(const Boson &, int) const;

        const std::vector<Real>& get_visible_bias() const;
        const std::vector<Real>& get_hidden_bias() const;
        const std::vector<std::vector<Real>>& get_weights() const;
};
inline Real RBMWavefunction::derivative_alpha(const System&) const {
    throw std::logic_error("Function not defined for RBM, only here for bad design reasons.");
}
inline Real RBMWavefunction::drift_force(const Boson &, int) const {
    throw std::logic_error("Function not implemented.");
}
inline const std::vector<Real>& RBMWavefunction::get_visible_bias() const {
    return _a;
}
inline const std::vector<Real>& RBMWavefunction::get_hidden_bias() const {
    return _b;
}
inline const std::vector<std::vector<Real>>& RBMWavefunction::get_weights() const {
    return _w;
}
