import math
from itertools import product

import numpy
from autograd import grad, hessian
from autograd import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from qflow.wavefunctions import SimpleGaussian, SumPooling

from .testutils import array_strat, assert_close, float_strat


def sum_pool_np(X, alpha):
    def psi(X):
        return np.exp(-alpha * np.sum(X ** 2))

    return np.sum(
        [(i != j) * psi(X[(i, j), :]) for i in range(len(X)) for j in range(len(X))]
    )


@given(X=array_strat(max_size=50), alpha=st.floats(min_value=0, allow_infinity=False))
@settings(deadline=None)
def test_eval(X, alpha):
    psi = SimpleGaussian(alpha)
    pool = SumPooling(psi)
    numpy.testing.assert_allclose(sum_pool_np(X, alpha), pool(X))


@given(X=array_strat(max_size=20), alpha=st.floats(min_value=0, allow_infinity=False))
@settings(deadline=None)
def test_gradient(X, alpha):
    psi = SimpleGaussian(alpha)
    pool = SumPooling(psi)

    assert_close(pool.gradient(X)[1], 0)
    assert_close(
        [grad(sum_pool_np, 1)(X, alpha) / sum_pool_np(X, alpha)], pool.gradient(X)[:1]
    )


@given(X=array_strat(max_size=10), alpha=st.floats(min_value=0, max_value=10))
@settings(deadline=None)
def test_drift_force(X, alpha):
    psi = SimpleGaussian(alpha)
    pool = SumPooling(psi)

    expected = 2 * grad(sum_pool_np, 0)(X, alpha) / sum_pool_np(X, alpha)
    assert_close(expected.ravel(), pool.drift_force(X))


@given(X=array_strat(max_size=10), alpha=st.floats(min_value=0, max_value=10))
@settings(deadline=None)
def test_laplacian(X, alpha):
    psi = SimpleGaussian(alpha)
    pool = SumPooling(psi)

    expected = np.trace(
        hessian(sum_pool_np)(X, alpha).reshape(X.size, X.size)
    ) / sum_pool_np(X, alpha)

    assert_close(expected, pool.laplacian(X))
