import unittest

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from qflow.samplers import MetropolisSampler
from qflow.wavefunctions import RBMSymmetricWavefunction as SRBM
from qflow.wavefunctions import RBMWavefunction as RBM
from qflow.wavefunctions import SimpleGaussian


class TestSymmetryMetric(unittest.TestCase):
    @given(
        P=st.integers(min_value=1, max_value=20),
        D=st.integers(min_value=1, max_value=3),
        alpha=st.floats(min_value=0, max_value=100),
    )
    def test_gaussian_is_symmetric(self, P, D, alpha):
        psi = SimpleGaussian(alpha)
        sampler = MetropolisSampler(np.empty((P, D)), psi, 0.1)
        s = psi.symmetry_metric(sampler, 10)
        self.assertAlmostEqual(1, s, places=9)

    @given(
        P=st.integers(min_value=1, max_value=20),
        D=st.integers(min_value=1, max_value=3),
        N=st.integers(min_value=1, max_value=20),
        sigma2=st.floats(min_value=0.1, max_value=100),
    )
    def test_symmetric_rbm_is_symmetric(self, P, D, N, sigma2):
        psi = SRBM(P * D, N, D, sigma2=sigma2)
        sampler = MetropolisSampler(np.empty((P, D)), psi, 0.1)
        s = psi.symmetry_metric(sampler, 10)
        self.assertAlmostEqual(1, s, places=9)
