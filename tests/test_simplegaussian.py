from autograd import grad, hessian
from autograd import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays

from qflow.wavefunctions import SimpleGaussian

from .testutils import assert_close, float_strat, array_strat


def psi_np(X, alpha, beta):
    coefs = (1, 1, beta)[: X.shape[1]]
    return np.exp(-alpha * np.sum(np.dot(X ** 2, coefs)))


@given(X=array_strat(), alpha=float_strat(), beta=float_strat())
def test_eval(X, alpha, beta):
    psi = SimpleGaussian(alpha, beta)
    assert_close(psi_np(X, alpha, beta), psi(X))


@given(X=array_strat(), alpha=float_strat(), beta=float_strat())
def test_gradient(X, alpha, beta):
    psi = SimpleGaussian(alpha, beta)

    gradient = psi.gradient(X)
    assert len(gradient) == 2
    assert gradient[1] == 0  # Beta fixed.
    assert_close(grad(psi_np, 1)(X, alpha, beta), gradient[0] * psi(X))


@given(X=array_strat(), alpha=float_strat(), beta=float_strat())
def test_drift_force(X, alpha, beta):
    psi = SimpleGaussian(alpha, beta)

    drift = 0.5 * psi.drift_force(X) * psi(X)
    assert len(drift) == X.size
    assert_close(grad(psi_np, 0)(X, alpha, beta).ravel(), drift)


@given(X=array_strat(), alpha=float_strat(), beta=float_strat())
def test_laplacian(X, alpha, beta):
    psi = SimpleGaussian(alpha, beta)

    laplacian = psi.laplacian(X) * psi(X)
    assert_close(
        np.trace(hessian(psi_np)(X, alpha, beta).reshape(X.size, X.size)), laplacian
    )
