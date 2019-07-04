import numpy as np
import pytest

from qflow.hamiltonians import HarmonicOscillator
from qflow.samplers import ImportanceSampler
from qflow.wavefunctions import SimpleGaussian
from qflow import DistanceCache

H0 = HarmonicOscillator()
psi0 = SimpleGaussian(0.5)

small_system = np.zeros((2, 2))
large_system = np.zeros((50, 3))
samples = 10000


def local_energy_gradient(H, psi, sampler, samples):
    return H.local_energy_gradient(sampler, psi, samples)


@pytest.mark.benchmark(group="Small system", warmup=True)
def test_E_L_small(benchmark):
    result = benchmark(
        H0.local_energy, ImportanceSampler(small_system, psi0), psi0, samples
    )
    assert np.isfinite(result).all()


@pytest.mark.benchmark(group="Small system", warmup=True)
def test_E_L_grad_small(benchmark):
    result = benchmark(
        H0.local_energy_gradient, ImportanceSampler(small_system, psi0), psi0, samples
    )
    assert np.isfinite(result).all()


@pytest.mark.benchmark(group="Small system", warmup=True)
def test_mean_dist_small(benchmark):
    result = benchmark(H0.mean_distance_array, ImportanceSampler(small_system, psi0), samples)
    assert np.isfinite(result).all()


@pytest.mark.benchmark(group="Large system", warmup=True)
def test_E_L_large(benchmark):
    result = benchmark(
        H0.local_energy, ImportanceSampler(large_system, psi0), psi0, samples
    )
    assert np.isfinite(result).all()


@pytest.mark.benchmark(group="Large system", warmup=True)
def test_E_L_grad_large(benchmark):
    result = benchmark(
        H0.local_energy_gradient, ImportanceSampler(large_system, psi0), psi0, samples
    )
    assert np.isfinite(result).all()


@pytest.mark.benchmark(group="Large system", warmup=True)
def test_mean_dist_large(benchmark):
    result = benchmark(H0.mean_distance_array, ImportanceSampler(large_system, psi0), samples)
    assert np.isfinite(result).all()


@pytest.mark.benchmark(group="Large system", warmup=True)
def test_mean_dist_large_cached(benchmark):
    with DistanceCache(large_system):
        result = benchmark(
            H0.mean_distance_array, ImportanceSampler(large_system, psi0), samples
        )
        assert np.isfinite(result).all()
