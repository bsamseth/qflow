import numpy as np
import pytest

from qflow.hamiltonians import HarmonicOscillator
from qflow.samplers import ImportanceSampler
from qflow.wavefunctions import SimpleGaussian, Dnn
from qflow.wavefunctions.nn.layers import DenseLayer
from qflow.wavefunctions.nn.activations import (
    sigmoid,
    tanh,
    relu,
    identity,
    exponential,
)
from qflow import DistanceCache

small_system = np.zeros((2, 2))
large_system = np.zeros((50, 3))
samples = 10000

H0 = HarmonicOscillator()
psi0 = SimpleGaussian(0.5)
layers = [
    DenseLayer(50 * 3, 32, activation=tanh, scale_factor=0.001),
    DenseLayer(32, 16, activation=tanh),
    DenseLayer(16, 1, activation=exponential),
]
dnn = Dnn()
for l in layers:
    dnn.add_layer(l)


def local_energy_gradient(H, psi, sampler, samples):
    return H.local_energy_gradient(sampler, psi, samples)


@pytest.mark.benchmark(group="evaluation", warmup=True)
def test_dnn_eval(benchmark):
    benchmark(dnn, large_system)


@pytest.mark.benchmark(group="gradient", warmup=True)
def test_dnn_gradient(benchmark):
    benchmark(dnn.gradient, large_system)


@pytest.mark.benchmark(group="laplacian", warmup=True)
def test_dnn_laplacian(benchmark):
    benchmark(dnn.laplacian, large_system)
