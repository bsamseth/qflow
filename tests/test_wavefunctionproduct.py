import unittest
import numpy as np

from qflow.wavefunction import SimpleGaussian, WavefunctionProduct


class TestWavefunctionProduct(unittest.TestCase):
    def test_works_on_simple_gaussians(self):
        """
        If
            Psi = SimpleGaussian(a) * SimpleGaussian(b)

        then we have by the form of SimpleGaussian the follwing

            Psi = SimpleGaussian(a + b)

        for any a and b.
        """
        for _ in range(1000):
            alpha1, alpha2 = np.random.rand(2)
            psi1 = SimpleGaussian(alpha1)
            psi2 = SimpleGaussian(alpha2)
            psi_expected = SimpleGaussian(alpha1 + alpha2)
            psi_prod = WavefunctionProduct(psi1, psi2)

            n, d = np.random.randint(100), np.random.randint(1, 3 + 1)
            s = np.random.randn(n, d)
            self.assertAlmostEqual(psi_expected(s), psi_prod(s))
            self.assertAlmostEqual(psi_expected.laplacian(s), psi_prod.laplacian(s))
            np.testing.assert_allclose(
                psi_expected.drift_force(s), psi_prod.drift_force(s)
            )

            # The gradient will be slightly different. Both psi1 and psi2 should give the same gradient, as it is
            # independent of alpha. However, the product state will give the gradients from both psi1 and psi2,
            # which means it will have two sets of each number.
            # __This is the expected behaviour.__
            expected_grad = psi_expected.gradient(s)
            np.testing.assert_allclose(
                np.concatenate([expected_grad] * 2), psi_prod.gradient(s)
            )
