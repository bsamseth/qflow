#pragma once

#include "definitions.hpp"
#include "hamiltonian.hpp"
#include "optimizer.hpp"
#include "sampler.hpp"
#include "system.hpp"
#include "wavefunction.hpp"

#include <algorithm>
#include <cassert>
#include <functional>
#include <vector>

/**
 * Class for representing a wavefunction with an RBM.
 */
class RBMWavefunction : public Wavefunction
{
public:
    // Attributes set as public for easy access/modification.
    const Real _sigma2;       // The square of $\sigma$.
    const Real _root_factor;  // Factor used to change representation, psi -> sqrt(psi).
    const int  _M;            // Number of visible nodes.
    const int  _N;            // Number of hidden nodes.

    constexpr static Real GIBBS_FACTOR
        = 0.5;  // Setting _root_factor to this will result in
                // the RBM suitable for Gibbs sampling.

    // The parameter vector is defined in the superclass.

    /**
     * Construct a wavefunction RBM.
     * @param M number of visible nodes.
     * @param N number of hidden nodes.
     * @param sigma2 square of the spread parameter $\sigma$.
     * @param root_factor default to 1, or set to `GIBBS_FACTOR` to get sqrt(psi).
     */
    RBMWavefunction(int M, int N, Real sigma2 = 1, Real root_factor = 1);

    /**
     * @param system configuration to evaluate the gradient for.
     * @return Gradient of the expected local energy, wrt. all the parameters.
     */
    RowVector gradient(const System& system) override;

    /**
     * Evaluate the RBM.
     * @param system configuration to evaluate for.
     * @return Explicit value of the wavefunction, as given by the RBM.
     */
    Real operator()(const System& system) override;

    /**
     * @param k bias index.
     * @param system configuration to evaluate for.
     * @return 1/psi d(psi)/d(a_k)
     */
    virtual Real deriv_a(int k, const System& system) const;

    /**
     * @param k bias index.
     * @param system configuration to evaluate for.
     * @return 1/psi d(psi)/d(b_k)
     */
    Real deriv_b(int k, const System& system) const;

    /**
     * @param k first weight index.
     * @param l second weight index.
     * @param system configuration to evaluate for.
     * @return 1/psi d(psi)/d(w_kl)
     */
    virtual Real deriv_w(int k, int l, const System& system) const;

    /**
     * @param system configuration to evaluate for.
     * @return 1/psi laplacian(psi)
     */
    Real laplacian(const System& system) override;

    /**
     * Return the drift force, as used in importance sampling.
     * @param system configuration to evaluate for.
     * @param particle_index which particle to evaluate the force for.
     * @param dim_index which dimension to evaluate the force for.
     * @return 2 grad(psi) / psi.
     */
    Real drift_force(const System& system, int particle_index, int dim_index) override;

    // Helpers. Public so that they are visible to the tests.
    Real v_j(int j, const System& system) const;
    // The following functions return the index into the full parameter vector which
    // point to the specified parameter.
    virtual int a(int i) const;
    virtual int b(int j) const;
    virtual int w(int i, int j) const;
};

inline int RBMWavefunction::a(int i) const
{
    assert(i >= 0 and i < _M);
    return i;
}
inline int RBMWavefunction::b(int j) const
{
    assert(j >= 0 and j < _N);
    return _M + j;
}
inline int RBMWavefunction::w(int i, int j) const
{
    assert(i >= 0 and i < _M);
    assert(j >= 0 and j < _N);
    return _M + _N + i * _N + j;
}
