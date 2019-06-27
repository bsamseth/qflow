import math
import warnings

import numpy
from autograd import grad, hessian
from autograd import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from qflow.wavefunctions import JastrowPade

from .testutils import array_strat, float_strat


def jastrow_np(X, alpha, beta):
    exponent = 0
    for i in range(X.shape[0]):
        for j in range(i + 1, X.shape[0]):
            r_ij = np.dot(X[i] - X[j], X[i] - X[j]) ** 0.5
            exponent += alpha * r_ij / (1 + beta * r_ij)
    return np.exp(exponent)


@given(X=array_strat(min_dims=2), beta=float_strat())
def test_eval(X, beta):

    psi = JastrowPade(0.5, beta)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert np.isclose(jastrow_np(X, 0.5, beta), psi(X))


@given(X=array_strat(min_dims=2), alpha=float_strat(), beta=float_strat())
@settings(deadline=None)
def test_gradient(X, alpha, beta):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np_grad_alpha = grad(jastrow_np, 1)(X, alpha, beta) / jastrow_np(X, alpha, beta)
        np_grad_beta = grad(jastrow_np, 2)(X, alpha, beta) / jastrow_np(X, alpha, beta)

    psi_alpha_const = JastrowPade(alpha, beta)
    psi = JastrowPade(alpha, beta, False)
    actual = psi.gradient(X)
    assert 2 == len(actual)

    assert psi_alpha_const.gradient(X)[0] == 0

    if math.isfinite(np_grad_beta):
        assert np.isclose(np_grad_alpha, actual[0])
        assert np.isclose(np_grad_beta, actual[1])


@given(X=array_strat(min_dims=2, max_size=10), beta=float_strat())
@settings(deadline=None)
def test_drift_force(X, beta):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np_drift = 2 * grad(jastrow_np, 0)(X, 0.5, beta) / jastrow_np(X, 0.5, beta)

    psi = JastrowPade(0.5, beta)
    for expect, actual in zip(np_drift.ravel(), psi.drift_force(X)):
        if math.isfinite(expect):
            assert np.isclose(expect, actual)


# Hessian calculation is super slow, limit size of inputs.
@given(X=array_strat(min_dims=2, max_size=5), beta=float_strat())
@settings(deadline=None)
def test_laplace(X, beta):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np_expect = np.trace(
            hessian(jastrow_np)(X, 0.5, beta).reshape(X.size, X.size)
        ) / jastrow_np(X, 0.5, beta)

    psi = JastrowPade(0.5, beta)
    if math.isfinite(np_expect):
        assert numpy.isclose(np_expect, psi.laplacian(X))
