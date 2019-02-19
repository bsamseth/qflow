import unittest
import warnings

from autograd import numpy as auto_np
from autograd import elementwise_grad, hessian
import numpy as np

from qflow.wavefunctions import Dnn
from qflow.layers import DenseLayer, InputLayer
from qflow.layers.activations import sigmoid, relu


def sigmoid_np(x):
    return 1 / (1 + auto_np.exp(-x))


def sigmoid_deriv(u):
    return u * (1 - u)


def relu_np(x):
    return auto_np.maximum(0, x)


def relu_deriv(y):
    return auto_np.where(y > 0, 1, 0)


class TestDnn(unittest.TestCase):
    def setUp(self):
        auto_np.random.seed(1234)
        self.nn = Dnn()
        self.input_layer = DenseLayer(2, 3, activation=sigmoid)
        self.middle_layer = DenseLayer(3, 4, activation=relu)
        self.output_layer = DenseLayer(4, 1)
        self.nn.add_layer(self.input_layer)
        self.nn.add_layer(self.middle_layer)
        self.nn.add_layer(self.output_layer)

        self.W1 = self.nn.layers[0].weights
        self.b1 = self.nn.layers[0].biases
        self.W2 = self.nn.layers[1].weights
        self.b2 = self.nn.layers[1].biases
        self.W3 = self.nn.layers[2].weights
        self.b3 = self.nn.layers[2].biases

        self.params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

        def f(x, w1, b1, w2, b2, w3, b3):
            z1 = x @ w1 + b1
            z2 = sigmoid_np(z1) @ w2 + b2
            z3 = relu_np(z2) @ w3 + b3
            return z3

        self.f_np = f

    def test_evaluate(self):
        for _ in range(10):
            for x in auto_np.random.randn(500, 2):
                np.testing.assert_almost_equal(self.f_np(x, *self.params), self.nn(x))

    def test_parameter_gradients(self):
        for _ in range(10):
            x = auto_np.random.randn(1, 2)
            output = self.nn(x)
            grads = self.nn.gradient(x)

            # Compute gradients using autograd
            auto_grads = []
            for i in range(len(self.nn.layers)):
                W_grad = elementwise_grad(self.f_np, 1 + 2 * i)(x, *self.params)
                b_grad = elementwise_grad(self.f_np, 1 + 2 * i + 1)(x, *self.params)

                auto_grads.extend(W_grad.ravel())
                auto_grads.extend(b_grad.ravel())

                # Test each indivudual layer
                np.testing.assert_almost_equal(
                    W_grad, self.nn.layers[i].weights_gradient
                )
                np.testing.assert_almost_equal(b_grad, self.nn.layers[i].bias_gradient)

            # Test extraction of full gradient vector
            np.testing.assert_almost_equal(auto_grads, grads * output)

    def test_drift_force(self):
        for _ in range(10):
            x = auto_np.random.randn(1, 2)

            # Autograd computes gradient per row of x. Want the sum over rows.
            auto_gradient = np.sum(
                elementwise_grad(self.f_np, 0)(x, *self.params), axis=0
            )

            np.testing.assert_almost_equal(
                auto_gradient, self.nn.drift_force(x) * self.nn(x) / 2
            )

    def test_laplace(self):
        # Autograd makes some warnings about code that is not ours. Ignore them here.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)

            hess = hessian(self.f_np)
            for _ in range(10):
                x = auto_np.random.randn(1, 2)  # Autograd hessian slow, less testing.
                output = self.nn(x)

                # Need to feed autograd hessian one row at a time and sum results.
                expected = sum(
                    np.trace(hess(x[i], *self.params)[0]) for i in range(x.shape[0])
                )

                self.assertAlmostEqual(expected, self.nn.laplacian(x) * output)
