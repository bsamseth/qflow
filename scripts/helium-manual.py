import sys
import time
import numpy as np
from datetime import timedelta
import pprint

from qflow.mpi import mpiprint
from qflow.hamiltonians import LennardJones
from qflow.samplers import HeliumSampler
from qflow.statistics import compute_statistics_for_series
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
P, D = 4, 3  # Particles, dimensions
L = (P / rho) ** (1 / 3)
system = np.empty((P, D))

mpiprint(f"P = {P}, D = {D}, rho = {rho}, L = {L}")

H = LennardJones(L)
mcmillian = JastrowMcMillian(5, 2.952, L)
layers = [
    DenseLayer(2 * D, 16, activation=tanh, scale_factor=0.005),
    DenseLayer(16, 1, activation=exponential),
]
dnn = Dnn()
for l in layers:
    dnn.add_layer(l)
mcmillian_fixed = FixedWavefunction(mcmillian)
psi = WavefunctionProduct(mcmillian_fixed, dnn)
# psi = mcmillian
sampler = HeliumSampler(system, psi, 0.5, L)
sampler.thermalize(5000)


points = 2 ** int(sys.argv[2])
t0 = time.time()
psi.symmetry_metric(sampler, 500)
t1 = time.time() - t0
eta = timedelta(seconds=round(t1 / 500 * points))
mpiprint(f"Calculating symmetry - ETA {eta}")
mpiprint(f"Symmetry: {psi.symmetry_metric(sampler, points)}")

t0 = time.time()
[H.local_energy(sampler.next_configuration(), psi) / P for _ in range(500)]
t1 = time.time() - t0
eta = timedelta(seconds=round(t1 / 500 * points))
mpiprint(f"Calculating final energy - ETA {eta}")

energies = [
    H.local_energy(sampler.next_configuration(), psi) / P for _ in range(points)
]
pprint.pprint(compute_statistics_for_series(energies, method="blocking"))
