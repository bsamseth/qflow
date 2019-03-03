import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from qflow.samplers import MetropolisSampler
from qflow.wavefunctions import RBMSymmetricWavefunction as SRBM
from qflow.wavefunctions import SimpleGaussian


@given(
    P=st.integers(min_value=1, max_value=20),
    D=st.integers(min_value=1, max_value=3),
    alpha=st.floats(min_value=0, max_value=100),
)
@settings(deadline=None)
def test_gaussian_is_symmetric(P, D, alpha):
    psi = SimpleGaussian(alpha)
    sampler = MetropolisSampler(np.empty((P, D)), psi, 0.1)
    s = psi.symmetry_metric(sampler, 10)
    assert np.isclose(1, s)


@given(
    P=st.integers(min_value=1, max_value=20),
    D=st.integers(min_value=1, max_value=3),
    N=st.integers(min_value=1, max_value=20),
    sigma2=st.floats(min_value=0.1, max_value=100),
)
@settings(deadline=None)
def test_symmetric_rbm_is_symmetric(P, D, N, sigma2):
    psi = SRBM(P * D, N, D, sigma2=sigma2)
    sampler = MetropolisSampler(np.empty((P, D)), psi, 0.1)
    s = psi.symmetry_metric(sampler, 10)
    assert np.isclose(1, s)
