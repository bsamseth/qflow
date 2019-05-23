import sys
import os
import time
import numpy as np
from datetime import datetime, timedelta
import pprint
from mpi4py import MPI
from tqdm import trange

from qflow.mpi import mpiprint
from qflow.hamiltonians import LennardJones
from qflow.optimizers import AdamOptimizer, SgdOptimizer
from qflow.samplers import HeliumSampler
from qflow.statistics import compute_statistics_for_series
from qflow.training import EnergyCallback, ParameterCallback, SymmetryCallback, train
from qflow.wavefunctions import (
    Dnn,
    FixedWavefunction,
    JastrowMcMillian,
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

rho = 0.365 / (2.556) ** 3  # Å^-3
P, D = 32, 3  # Particles, dimensions
L = (P / rho) ** (1 / 3)
system = np.empty((P, D))

mpiprint(f"P = {P}, D = {D}, rho = {rho}, L = {L}")

H = LennardJones(L)
mcmillian = JastrowMcMillian(5, 2.9, L)
layers = [
    DenseLayer(P * D, 128, activation=tanh, scale_factor=0.0005),
    DenseLayer(128, 64, activation=tanh),
    DenseLayer(64, 1, activation=exponential),
]
dnn = Dnn()
for l in layers:
    dnn.add_layer(l)
psi = WavefunctionProduct(mcmillian, dnn)
psi.parameters = np.loadtxt(sys.argv[1], delimiter=',')
# psi = mcmillian
sampler = HeliumSampler(system, psi, 1, L)
sampler.thermalize(5000)


points = 2 ** int(sys.argv[2])
t0 = time.time()
[H.local_energy(sampler.next_configuration(), psi) / P for _ in range(500)]
t1 = time.time() - t0
eta = timedelta(seconds=round(t1 / 500 * points))
mpiprint(f"Calculating final energy - ETA {eta}")

energies = [
    H.local_energy(sampler.next_configuration(), psi) / P for _ in range(points)
]
pprint.pprint(compute_statistics_for_series(energies, method="blocking"))
