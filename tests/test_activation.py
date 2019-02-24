import unittest

import numpy as np

from qflow.layers.activations import exponential, identity, relu, sigmoid, tanh


class TestDnn(unittest.TestCase):
    def setUp(self):
        pass

    def test_identity(self):
        for rows, cols in np.random.randint(1, 100, size=(500, 2)):
            x = np.random.randn(rows, cols)

            np.testing.assert_array_equal(x, identity.evaluate(x))
            np.testing.assert_array_equal(np.ones_like(x), identity.derivative(x))
            np.testing.assert_array_equal(np.zeros_like(x), identity.dbl_derivative(x))

    def test_relu(self):
        for rows, cols in np.random.randint(1, 100, size=(500, 2)):
            x = np.random.randn(rows, cols)

            np.testing.assert_array_equal(np.where(x > 0, x, 0), relu.evaluate(x))
            np.testing.assert_array_equal(
                np.where(x > 0, 1, 0), relu.derivative(relu.evaluate(x))
            )
            np.testing.assert_array_equal(
                np.zeros_like(x), relu.dbl_derivative(relu.evaluate(x))
            )

    def test_sigmoid(self):
        for rows, cols in np.random.randint(1, 100, size=(500, 2)):
            x = np.random.randn(rows, cols)
            sig = 1 / (1 + np.exp(-x))

            np.testing.assert_array_equal(sig, sigmoid.evaluate(x))
            np.testing.assert_array_equal(sig * (1 - sig), sigmoid.derivative(sig))
            np.testing.assert_array_equal(
                sig * (1 - sig) * (1 - 2 * sig), sigmoid.dbl_derivative(sig)
            )

    def test_tanh(self):
        for rows, cols in np.random.randint(1, 100, size=(50, 2)):
            x = np.random.randn(rows, cols)
            ta = np.tanh(x)

            np.testing.assert_array_equal(ta, tanh.evaluate(x))
            np.testing.assert_array_equal(
                1 - ta ** 2, tanh.derivative(tanh.evaluate(x))
            )
            np.testing.assert_allclose(
                -2 * np.sinh(x) / np.cosh(x) ** 3,
                tanh.dbl_derivative(tanh.evaluate(x)),
                rtol=1e-12,
            )

    def test_exponential(self):
        for rows, cols in np.random.randint(1, 100, size=(50, 2)):
            x = np.random.randn(rows, cols)
            exp = np.exp(x)

            np.testing.assert_array_equal(exp, exponential.evaluate(x))
            np.testing.assert_array_equal(
                exp, exponential.derivative(exponential.evaluate(x))
            )
            np.testing.assert_array_equal(
                exp, exponential.dbl_derivative(exponential.evaluate(x))
            )
