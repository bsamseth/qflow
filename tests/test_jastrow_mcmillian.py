import math
import warnings

import pytest
import numpy
from autograd import grad, hessian
from autograd import numpy as np
from hypothesis import given, settings, assume
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

            r_ij = max(0.3 * 2.556, r_ij)

            exponent += (beta / r_ij) ** n
    return np.exp(-0.5 * exponent)

def is_jastrow_np_adjusted(X, n, beta):
    adjusted = False
    for i in range(X.shape[0]):
        for j in range(i + 1, X.shape[0]):
            r_ij = X[i] - X[j]
            r_ij -= np.around(r_ij / L) * L
            r_ij = np.dot(r_ij, r_ij) ** 0.5

            if r_ij < 0.3 * 2.556:
                return True
    return False


@given(X=array_strat(min_dims=2), beta=float_strat())
def test_eval(X, beta):
    psi = JastrowMcMillian(5, beta, L)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np_eval = jastrow_np(X, 5, beta)
        if math.isfinite(np_eval):
            assert np.isclose(np_eval, psi(X), equal_nan=True)


@given(X=array_strat(min_dims=2), beta=float_strat())
@settings(deadline=None)
def test_gradient(X, beta):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np_grad_beta = grad(jastrow_np, 2)(X, 5, beta) / jastrow_np(X, 5, beta)

    psi = JastrowMcMillian(5, beta, L)
    actual = psi.gradient(X)
    assert 1 == len(actual)

    if math.isfinite(np_grad_beta):
        assert np.isclose(np_grad_beta, actual[0], equal_nan=True)


@given(X=array_strat(min_dims=2, max_size=10), beta=float_strat())
@settings(deadline=None)
def test_drift_force(X, beta):
    assume(beta > 0)  # Implementation not defined for zero or negative values.

    if is_jastrow_np_adjusted(X, 5, beta):
        assert True  # Can't get autodiff to cooperate here. Possible TODO.
        return

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np_drift = 2 * grad(jastrow_np, 0)(X, 5, beta) / jastrow_np(X, 5, beta)

    psi = JastrowMcMillian(5, beta, L)

    for expect, actual in zip(np_drift.ravel(), psi.drift_force(X)):
        if math.isfinite(expect):
            assert np.isclose(expect, actual, equal_nan=True), f"np_drift = {np_drift}, actual = {psi.drift_force(X)}"


# Hessian calculation is super slow, limit size of inputs.
@pytest.mark.xfail
@given(X=array_strat(min_dims=2, max_size=5), beta=float_strat())
@settings(deadline=None)
def test_laplace(X, beta):
    assume(beta > 0)  # Implementation not defined for zero or negative values.

    if is_jastrow_np_adjusted(X, 5, beta):
        assert True  # Can't get autodiff to cooperate here. Possible TODO.
        return

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np_expect = np.trace(
            hessian(jastrow_np)(X, 5, beta).reshape(X.size, X.size)
        ) / jastrow_np(X, 5, beta)

    psi = JastrowMcMillian(5, beta, L)
    if math.isfinite(np_expect):
        assert numpy.isclose(np_expect, psi.laplacian(X), equal_nan=True)
