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
    InputSorter,
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

os.makedirs("logfiles", exist_ok=True)

rho = 0.365 / (2.556) ** 3  # Å^-3
P, D = 32, 3  # Particles, dimensions
L = (P / rho) ** (1 / 3)
system = np.empty((P, D))

mpiprint(f"P = {P}, D = {D}, rho = {rho}, L = {L}")

H = LennardJones(L)
mcmillian = JastrowMcMillian(5, 2.965, L)
layers = [
    DenseLayer(P * D, 144, activation=tanh, scale_factor=0.001),
    DenseLayer(144, 36, activation=tanh),
    DenseLayer(36, 1, activation=exponential),
]
dnn = Dnn()
for l in layers:
    dnn.add_layer(l)
mcmillian_fixed = FixedWavefunction(mcmillian)
psi_total = WavefunctionProduct(mcmillian_fixed, dnn)
psi = InputSorter(psi_total)
# psi = mcmillian
sampler = HeliumSampler(system, psi, 0.5, L)
sampler.thermalize(10000)
mpiprint(f"AR: {sampler.acceptance_rate}")

optimizer = AdamOptimizer(len(psi.parameters))


steps = 1000
t_average = 0
E_training = []
b_training = []
for _ in range(steps):
    t0 = time.time()
    H.optimize_wavefunction(psi, sampler, 50, 1000, optimizer, 0, False)
    t1 = time.time() - t0
    t_average = (t_average * _ + t1) / (_ + 1)
    E = H.local_energy_array(sampler, psi, 5000) / P
    E_training.append(np.mean(E))
    b_training.append(psi.parameters[0])
    eta = timedelta(seconds=round(t_average * (steps - _)))
    mpiprint(
        f"Step {_+1:5d}/{steps:d} - {1 / t1:5.3f} it/s - ETA {eta} - AR = {sampler.acceptance_rate:.4f} - " +
        f"<E> = {np.mean(E_training):3.5f} ({E_training[-1]:3.5f}) - " +
        f"E_sem = {np.std(E_training) / np.sqrt(len(E_training)):3.3f}  - " +
        f"params[0] = {np.mean(b_training):3.5f} ({b_training[-1]:3.5f})"
    )
    if MPI.COMM_WORLD.rank == 0:
        np.savetxt(
            f"logfiles/helium-InputSorterDnn-P{P}-D{D}-{_:06d}-parameters.csv",
            psi.parameters,
            delimiter=",",
        )
        np.savetxt(
            f"logfiles/helium-InputSorterDnn-P{P}-D{D}-training-energies.csv",
            E_training,
            delimiter=",",
        )

# psi.parameters = np.mean(b_training[-steps//10:], keepdims=True)

points = 2 ** 21
t0 = time.time()
H.local_energy_array(sampler, psi, 500)
t1 = time.time() - t0
eta = timedelta(seconds=round(t1 / 500 * points))
mpiprint(f"Calculating final energy - ETA {eta}")

energies = np.array(H.local_energy_array(sampler, psi, points)) / P

if MPI.COMM_WORLD.rank == 0:
    np.savetxt(
        f"logfiles/helium-InputSorterDnn-P{P}-D{D}-energies.csv", energies, delimiter=","
    )
    pprint.pprint(compute_statistics_for_series(energies, method="blocking"))
