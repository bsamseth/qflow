#pragma once

#include "rbmwavefunction.hpp"

#include <algorithm>
#include <cassert>
#include <functional>
#include <vector>

/**
 * Class for representing a wavefunction with an symmetric RBM.
 */
class RBMSymmetricWavefunction : public RBMWavefunction
{
public:
    const int _f;  // Number of degrees of freedom per particle.

    // The parameter vector is defined in the superclass.

    /**
     * Construct a wavefunction RBM.
     * @param M number of visible nodes.
     * @param N number of hidden nodes.
     * @param sigma2 square of the spread parameter $\sigma$.
     * @param root_factor default to 1, or set to `GIBBS_FACTOR` to get sqrt(psi).
     */
    RBMSymmetricWavefunction(int  M,
                             int  N,
                             int  f,
                             Real sigma2      = 1,
                             Real root_factor = 1);

    /**
     * @param k bias index.
     * @param system configuration to evaluate for.
     * @return 1/psi d(psi)/d(a_k)
     */
    Real deriv_a(int k, const System& system) const override;

    /**
     * @param k first weight index.
     * @param l second weight index.
     * @param system configuration to evaluate for.
     * @return 1/psi d(psi)/d(w_kl)
     */
    Real deriv_w(int k, int l, const System& system) const override;

    /**
     * @param system configuration to evaluate the gradient for.
     * @return Gradient of the expected local energy, wrt. all the parameters.
     */
    RowVector gradient(const System& system) override;

    // The following functions return the index into the full parameter vector which
    // point to the specified parameter.
    int a(int i) const override;
    int b(int j) const override;
    int w(int i, int j) const override;
};

inline int RBMSymmetricWavefunction::a(int i) const
{
    assert(i >= 0 and i < _M);
    return i % _f;
}
inline int RBMSymmetricWavefunction::b(int j) const
{
    assert(j >= 0 and j < _N);
    return _f + j;
}
inline int RBMSymmetricWavefunction::w(int i, int j) const
{
    assert(i >= 0 and i < _M);
    assert(j >= 0 and j < _N);
    return _f + _N + (i % _f) * _N + j;
}
