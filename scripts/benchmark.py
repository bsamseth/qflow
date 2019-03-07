from timeit import timeit
from mpi4py import MPI

setup = """
import numpy as np
import matplotlib.pyplot as plt

from qflow.wavefunctions import (
    JastrowOrion,
    SimpleGaussian,
    WavefunctionProduct,
    FixedWavefunction,
    Dnn,
    SumPooling,
)
from qflow.wavefunctions.nn.layers import DenseLayer
from qflow.wavefunctions.nn.activations import sigmoid, tanh, relu, identity, exponential

from qflow.hamiltonians import (
    HarmonicOscillator,
    CoulombHarmonicOscillator,
)

from qflow.samplers import MetropolisSampler, ImportanceSampler
from qflow.optimizers import AdamOptimizer, SgdOptimizer
from qflow.training import train, EnergyCallback, SymmetryCallback, ParameterCallback
from qflow.statistics import compute_statistics_for_series


H1 = CoulombHarmonicOscillator()
P, D = 2, 2  # Particles, dimensions
N = 4  # Hidden nodes
system = np.empty((P, D))

simple_gaussian = SimpleGaussian(alpha=0.5)
jastrow = JastrowOrion(beta=2, gamma=1.5)
layers = [
    DenseLayer(2*D, 32, activation=tanh, scale_factor=0.001),
    DenseLayer(32, 16, activation=tanh),
    DenseLayer(16, 1, activation=exponential),
]
dnn = Dnn()
for l in layers:
    dnn.add_layer(l)

simple_and_jastrow = WavefunctionProduct(simple_gaussian, jastrow)
pooled_dnn = SumPooling(dnn)
psi = WavefunctionProduct(simple_and_jastrow, pooled_dnn)
psi_sampler = ImportanceSampler(system, psi, step_size=0.1)

s = psi_sampler.next_configuration()
"""

time = timeit("H1.local_energy_gradient(psi_sampler, psi, 10000)", setup=setup, number=20)
if MPI.COMM_WORLD.rank == 0:
    print('Benchmarking...', end='  ')
    print('total time:', time)


