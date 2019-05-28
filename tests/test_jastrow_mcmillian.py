import math
import warnings

import numpy
from autograd import grad, hessian
from autograd import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from qflow.wavefunctions import JastrowMcMillian

from .testutils import array_strat, float_strat

n_strat = lambda: st.integers(min_value=3, max_value=10)

L = 5  # Arbitrary simulation box size.

def jastrow_np(X, n, beta):
    exponent = 0
    for i in range(X.shape[0]):
        for j in range(i + 1, X.shape[0]):
            r_ij = X[i] - X[j]
            r_ij -= np.around(r_ij / L) * L
            r_ij = np.dot(r_ij, r_ij) ** 0.5

            if r_ij > 0.5 * L:
                continue

            r_ij = max(0.3 * 2.556, r_ij);

            exponent += (beta / r_ij) ** n
    return np.exp(-0.5 * exponent)


@given(X=array_strat(min_dims=2), n=n_strat(), beta=float_strat())
def test_eval(X, n, beta):

    psi = JastrowMcMillian(n, beta, L)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert np.isclose(jastrow_np(X, n, beta), psi(X), equal_nan=True)


@given(X=array_strat(min_dims=2), beta=float_strat())
@settings(deadline=None)
def test_gradient(X, beta):
    n=5
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np_grad_beta = grad(jastrow_np, 2)(X, n, beta) / jastrow_np(X, n, beta)

    psi = JastrowMcMillian(n, beta, L)
    actual = psi.gradient(X)
    assert 1 == len(actual)

    if math.isfinite(np_grad_beta):
        assert np.isclose(np_grad_beta, actual[0], equal_nan=True)


@given(X=array_strat(min_dims=2, max_size=10), n=n_strat(), beta=float_strat())
@settings(deadline=None)
def test_drift_force(X, n, beta):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np_drift = 2 * grad(jastrow_np, 0)(X, n, beta) / jastrow_np(X, n, beta)

    psi = JastrowMcMillian(n, beta, L)
    for expect, actual in zip(np_drift.ravel(), psi.drift_force(X)):
        if math.isfinite(expect):
            assert np.isclose(expect, actual, equal_nan=True)


# Hessian calculation is super slow, limit size of inputs.
@given(X=array_strat(min_dims=2, max_size=5), n=n_strat(), beta=float_strat())
@settings(deadline=None)
def test_laplace(X, n, beta):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np_expect = np.trace(
            hessian(jastrow_np)(X, n, beta).reshape(X.size, X.size)
        ) / jastrow_np(X, n, beta)

    psi = JastrowMcMillian(n, beta, L)
    if math.isfinite(np_expect):
        assert numpy.isclose(np_expect, psi.laplacian(X), equal_nan=True)
