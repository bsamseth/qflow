import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from qflow.wavefunctions.nn.activations import (
    exponential,
    identity,
    relu,
    sigmoid,
    tanh,
)

from .testutils import array_strat, assert_close


@given(array_strat(max_size=100))
def test_identity(x):
    np.testing.assert_array_equal(x, identity.evaluate(x))
    np.testing.assert_array_equal(np.ones_like(x), identity.derivative(x))
    np.testing.assert_array_equal(np.zeros_like(x), identity.dbl_derivative(x))


@given(array_strat(max_size=100))
def test_relu(x):
    np.testing.assert_array_equal(np.where(x > 0, x, 0), relu.evaluate(x))
    np.testing.assert_array_equal(
        np.where(x > 0, 1, 0), relu.derivative(relu.evaluate(x))
    )
    np.testing.assert_array_equal(
        np.zeros_like(x), relu.dbl_derivative(relu.evaluate(x))
    )


@given(array_strat(max_size=100))
def test_sigmoid(x):
    sig = 1 / (1 + np.exp(-x))

    np.testing.assert_array_equal(sig, sigmoid.evaluate(x))
    np.testing.assert_array_equal(sig * (1 - sig), sigmoid.derivative(sig))
    np.testing.assert_array_equal(
        sig * (1 - sig) * (1 - 2 * sig), sigmoid.dbl_derivative(sig)
    )


@given(array_strat(max_size=50))
def test_tanh(x):
    assume(np.all(np.abs(x) < 15))
    ta = np.tanh(x)

    assert_close(ta, tanh.evaluate(x))
    assert_close(1 - ta ** 2, tanh.derivative(tanh.evaluate(x)))
    assert_close(
        -2 * np.sinh(x) / np.cosh(x) ** 3, tanh.dbl_derivative(tanh.evaluate(x))
    )


@given(array_strat(max_size=100))
def test_exponential(x):
    exp = np.exp(x)

    np.testing.assert_array_equal(exp, exponential.evaluate(x))
    np.testing.assert_array_equal(exp, exponential.derivative(exponential.evaluate(x)))
    np.testing.assert_array_equal(
        exp, exponential.dbl_derivative(exponential.evaluate(x))
    )
