import math
import warnings

import numpy
from autograd import grad, hessian
from autograd import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from qflow.wavefunctions import JastrowOrion

from .testutils import array_strat, float_strat


def jastrow_np(X, beta, gamma):
    exponent = 0
    for i in range(X.shape[0]):
        for j in range(i + 1, X.shape[0]):
            r_ij2 = np.dot(X[i] - X[j], X[i] - X[j])
            exponent += -0.5 * beta ** 2 * r_ij2 + np.abs(beta * gamma) * np.sqrt(r_ij2)
    return np.exp(exponent)


@given(X=array_strat(min_dims=2), beta=float_strat(), gamma=float_strat())
def test_eval(X, beta, gamma):
    psi = JastrowOrion(beta, gamma)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert np.isclose(jastrow_np(X, beta, gamma), psi(X))


@given(X=array_strat(min_dims=2), beta=float_strat(), gamma=float_strat())
@settings(deadline=None)
def test_gradient(X, beta, gamma):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np_grad_beta = grad(jastrow_np, 1)(X, beta, gamma) / jastrow_np(X, beta, gamma)
        np_grad_gamma = grad(jastrow_np, 2)(X, beta, gamma) / jastrow_np(X, beta, gamma)

    psi = JastrowOrion(beta, gamma)
    actual = psi.gradient(X)
    assert 2 == len(actual)

    if math.isfinite(np_grad_beta):
        assert np.isclose(np_grad_beta, actual[0])
    if math.isfinite(np_grad_gamma):
        assert np.isclose(np_grad_gamma, actual[1])


@given(X=array_strat(min_dims=2, max_size=10), beta=float_strat(), gamma=float_strat())
@settings(deadline=None)
def test_drift_force(X, beta, gamma):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np_drift = 2 * grad(jastrow_np, 0)(X, beta, gamma) / jastrow_np(X, beta, gamma)

    psi = JastrowOrion(beta, gamma)
    for expect, actual in zip(np_drift.ravel(), psi.drift_force(X)):
        if math.isfinite(expect):
            assert np.isclose(expect, actual)


# Hessian calculation is super slow, limit size of inputs.
@given(X=array_strat(min_dims=2, max_size=5), beta=float_strat(), gamma=float_strat())
@settings(deadline=None)
def test_laplace(X, beta, gamma):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np_expect = np.trace(
            hessian(jastrow_np)(X, beta, gamma).reshape(X.size, X.size)
        ) / jastrow_np(X, beta, gamma)

    psi = JastrowOrion(beta, gamma)
    if math.isfinite(np_expect):
        assert numpy.isclose(np_expect, psi.laplacian(X))
