import os
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

from qflow.hamiltonians import CoulombHarmonicOscillator, HarmonicOscillator
from qflow.optimizers import AdamOptimizer, SgdOptimizer
from qflow.samplers import ImportanceSampler, MetropolisSampler
from qflow.statistics import compute_statistics_for_series
from qflow.training import EnergyCallback, ParameterCallback, SymmetryCallback, train
from qflow.wavefunctions import (
    Dnn,
    FixedWavefunction,
    JastrowOrion,
    SimpleGaussian,
    SumPooling,
    WavefunctionProduct,
)
from qflow.wavefunctions.nn.activations import (
    exponential,
    identity,
    relu,
    sigmoid,
    tanh,
)
from qflow.wavefunctions.nn.layers import DenseLayer

H1 = CoulombHarmonicOscillator()
P, D = 8, 3  # Particles, dimensions
N = 4  # Hidden nodes
system = np.empty((P, D))

simple_gaussian = SimpleGaussian(alpha=0.5)
jastrow = JastrowOrion(beta=2, gamma=1.5)
layers = [
    DenseLayer(2 * D, 32, activation=tanh, scale_factor=0.001),
    DenseLayer(32, 1, activation=exponential),
]
dnn = Dnn()
for l in layers:
    dnn.add_layer(l)

simple_and_jastrow = WavefunctionProduct(simple_gaussian, jastrow)
pooled_dnn = SumPooling(dnn)
psi = WavefunctionProduct(simple_and_jastrow, pooled_dnn)
psi_sampler = ImportanceSampler(system, psi, step_size=0.1)

psi_energies = EnergyCallback(samples=5000, verbose=True)
psi_symmetries = SymmetryCallback(samples=100)
psi_parameters = ParameterCallback()

train(
    psi,
    H1,
    psi_sampler,
    iters=50,
    samples=1000,
    gamma=0,
    optimizer=AdamOptimizer(len(psi.parameters)),
    # call_backs=(psi_energies, psi_symmetries, psi_parameters),
)

if MPI.COMM_WORLD.rank == 0:
    os.makedirs("logfiles", exist_ok=True)
    np.savetxt("logfiles/QD-run-energies.npz", psi_energies)
    np.savetxt("logfiles/QD-run-symmetries.npz", psi_symmetries)
    np.savetxt("logfiles/QD-run-parameters.npz", psi_energies)
