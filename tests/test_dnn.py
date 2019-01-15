import unittest
from autograd import numpy as auto_np
from autograd import elementwise_grad, hessian
import numpy as np

from EigenNN import Dnn
from EigenNN.layer import DenseLayer
from EigenNN.activation import sigmoid, relu


def sigmoid_np(x):
    return 1 / (1 + auto_np.exp(-x))


def sigmoid_deriv(u):
    return u * (1 - u)


def sigmoid_dbl_deriv(u):
    return u * (1 - u) * (1 - 2 * u)


def relu_np(x):
    return auto_np.maximum(0, x)

def relu_deriv(y):
    return auto_np.where(y > 0, 1, 0)

def relu_dbl_deriv(y):
    return y * 0


class TestDnn(unittest.TestCase):
    def setUp(self):
        auto_np.random.seed(1234)
        self.nn = Dnn()
        self.nn.add_layer(DenseLayer(2, 3, activation=sigmoid))
        self.nn.add_layer(DenseLayer(3, 4, activation=relu))
        self.nn.add_layer(DenseLayer(4, 1))

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
            x = auto_np.random.randn(500, 2)
            np.testing.assert_almost_equal(
                self.f_np(x, *self.params), self.nn.evaluate(x)
            )

    def test_parameter_gradients(self):
        for _ in range(10):
            x = auto_np.random.randn(500, 2)
            grads = self.nn.parameter_gradient(x)

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
            np.testing.assert_almost_equal(auto_grads, grads)

    def test_gradient(self):
        for _ in range(10):
            x = auto_np.random.randn(500, 2)

            # Autograd computes gradient per row of x. Want the sum over rows.
            auto_gradient = np.sum(
                elementwise_grad(self.f_np, 0)(x, *self.params), axis=0
            )

            np.testing.assert_almost_equal(auto_gradient, self.nn.gradient(x))

    def test_laplace(self):
        hess = hessian(self.f_np)
        for _ in range(10):
            x = auto_np.random.randn(50, 2)  # Autograd hessian slow, less testing.

            # Need to feed autograd hessian one row at a time and sum results.
            expected = sum(
                np.trace(hess(x[i], *self.params)[0]) for i in range(x.shape[0])
            )

            self.assertAlmostEqual(expected, self.nn.laplace(x))
