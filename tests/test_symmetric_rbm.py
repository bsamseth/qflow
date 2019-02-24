import unittest
import warnings

import numpy as np
from autograd import elementwise_grad, hessian
from autograd import numpy as auto_np

from qflow.wavefunctions import RBMSymmetricWavefunction as SRBM
from qflow.wavefunctions import RBMWavefunction as RBM

auto_np.random.seed(2019)


def sym_to_full(X, a, b, w):
    """Convert parameters from SRBM to RBM style"""
    X = X.flatten()
    f = w.size // b.size
    a = auto_np.repeat(a.reshape(1, -1), X.size // a.size, axis=0).flatten()
    w = auto_np.vstack([w] * (X.size // f))
    assert a.size == X.size
    assert w.shape == (len(a), len(b))
    return X, a, b, w


def rbm_np(X, a, b, w, sigma2=1):
    res = auto_np.exp(-auto_np.sum((X - a) ** 2) / (2 * sigma2))
    for j in range(b.size):
        res *= 1 + auto_np.exp(b[j] + auto_np.dot(X, w[:, j]) / sigma2)
    return res


def srbm_np(X, a, b, w, sigma2=1):
    f = a.size
    i = auto_np.arange(X.size)
    res = auto_np.exp(-auto_np.sum((X - a[i % f]) ** 2) / (2 * sigma2))
    for j in range(b.size):
        res *= 1 + auto_np.exp(b[j] + auto_np.dot(X, w[i % f, j]) / sigma2)
    return res


def randomized(test_func, size=10, max_dim=20):
    """Run a test function on randomized setups of SRBMs"""
    for P, D, N in np.random.randint(1, max_dim, size=(size, 3)):
        sigma2 = 1  # np.random.rand() * 5
        srbm = SRBM(P * D, N, D, sigma2=sigma2)
        a, b, w = (
            np.array(srbm.parameters[:D]),
            np.array(srbm.parameters[D : D + N]),
            np.array(srbm.parameters[D + N :]).reshape(D, N),
        )
        X = 0.1 * np.random.randn(P, D)

        # Rolling X along axis 0 will make particle i -> i + 1, with loopover.
        for shift in range(max(5, X.shape[0])):
            test_func(srbm, np.roll(X, shift, axis=0), a, b, w, sigma2)


class TestSRBM(unittest.TestCase):
    def test_evaluation_consistency(self):
        """
        Symmetric RBM should give same as RBM if RBM is given the same parameters,
        with biases and weights replicated.
        """

        def test(srbm, X, a, b, w, sigma2):
            # Setup regualr RBM with copied parameters to emulate symmetric RBM.
            (P, D), N = X.shape, len(b)
            rbm = RBM(P * D, N, sigma2=sigma2)
            X, a, b, w = sym_to_full(X, a, b, w)
            rbm.parameters = np.concatenate((a.flatten(), b.flatten(), w.flatten()))

            # Three different evaluations, should all be equal.
            # Equallity best tested comparing their ratio to unity. The default
            # tolerances used are tuned well for number expected close to unity.
            # Note also that the numpy true expression _cannot_ be zero.
            np_expect = rbm_np(X.flatten(), a, b, w, sigma2)
            self.assertAlmostEqual(1, rbm(X) / np_expect)
            self.assertAlmostEqual(1, srbm(X) / np_expect)

        randomized(test)

    def test_evaluation(self):
        def test(srbm, X, a, b, w, sigma2):
            # Check evaluation against the dedicated Symmetric RBM Numpy impl.
            # Do this for any permutation of the particles.
            np_expect = srbm_np(X.flatten(), a, b, w, sigma2=sigma2)

            self.assertAlmostEqual(1, srbm(X) / np_expect)

        randomized(test)

    def test_gradient(self):
        def test(srbm, X, a, b, w, sigma2):
            # Check evaluation against the dedicated Symmetric RBM Numpy impl.
            a_grad = elementwise_grad(srbm_np, 1)(X.flatten(), a, b, w, sigma2)
            b_grad = elementwise_grad(srbm_np, 2)(X.flatten(), a, b, w, sigma2)
            w_grad = elementwise_grad(srbm_np, 3)(X.flatten(), a, b, w, sigma2)

            # Remember that gradient returned is grad / eval
            np_expect = np.concatenate((a_grad, b_grad, w_grad.ravel())) / srbm_np(
                X.flatten(), a, b, w, sigma2
            )

            for expect, actual in zip(np_expect, srbm.gradient(X)):
                if expect == 0:
                    self.assertEqual(expect, actual)
                else:
                    self.assertAlmostEqual(1, actual / expect)

        randomized(test)

    def test_drift_force(self):
        def test(srbm, X, a, b, w, sigma2):
            np_expect = (
                2
                * elementwise_grad(srbm_np, 0)(X.flatten(), a, b, w, sigma2)
                / srbm_np(X.flatten(), a, b, w, sigma2)
            )

            for expect, actual in zip(np_expect, srbm.drift_force(X)):
                if expect == 0:
                    self.assertEqual(expect, actual)
                else:
                    self.assertAlmostEqual(1, actual / expect)

        randomized(test)

    def test_laplacian(self):
        def test(srbm, X, a, b, w, sigma2):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                np_expect = np.trace(
                    hessian(srbm_np)(X.flatten(), a, b, w, sigma2)
                ) / srbm_np(X.flatten(), a, b, w, sigma2)
                actual = srbm.laplacian(X)
                if np_expect == 0:
                    self.assertEqual(np_expect, actual)
                else:
                    self.assertAlmostEqual(1, actual / np_expect)

        # Hessian is slow, do smaller tests.
        randomized(test, size=5, max_dim=6)
