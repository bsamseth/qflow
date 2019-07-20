import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib2tikz
from datetime import timedelta
from mpi4py import MPI

from qflow.mpi import mpiprint
from qflow.hamiltonians import LennardJones
from qflow.optimizers import AdamOptimizer
from qflow.samplers import HeliumSampler
from qflow.statistics import compute_statistics_for_series, statistics_to_tex
from qflow.wavefunctions import Dnn, JastrowMcMillian, InputSorter, WavefunctionProduct
from qflow.wavefunctions.nn.activations import exponential, tanh
from qflow.wavefunctions.nn.layers import DenseLayer


def plot_training(energies, parameters):
    energies = np.asarray(energies)
    print(energies.shape)
    print(energies)
    _, (eax, pax) = plt.subplots(ncols=2)
    eax.plot(energies[:, 0], label=r"$\psi_{M}$")
    eax.plot(energies[:, 1], label=r"$\psi_{SDNN}$")
    eax.set_xlabel(r"% of training")
    eax.set_ylabel(r"Ground state energy [a.u]")
    eax.legend()

    pax.plot(np.asarray(parameters)[:, 1:50])
    pax.set_xlabel(r"% of training")

    matplotlib2tikz.save(__file__ + ".tex")


os.makedirs("logfiles", exist_ok=True)

rho = 0.365 / (2.556) ** 3  # Å^-3
P, D = 32, 3  # Particles, dimensions
L = (P / rho) ** (1 / 3)
system = np.empty((P, D))

mpiprint(f"P = {P}, D = {D}, rho = {rho}, L = {L}")

H = LennardJones(L)
layers = [
    DenseLayer(P * D, 144, activation=tanh, scale_factor=0.001),
    DenseLayer(144, 36, activation=tanh),
    DenseLayer(36, 1, activation=exponential),
]
dnn = Dnn()
for l in layers:
    dnn.add_layer(l)
mcmillian = JastrowMcMillian(5, 2.85, L)
psi_total = WavefunctionProduct(mcmillian, dnn)
psi = InputSorter(psi_total)
# psi = mcmillian
sampler = HeliumSampler(system, psi, 0.5, L)
sampler.thermalize(10000)
mpiprint(f"AR: {sampler.acceptance_rate}")

mcmillian_bench = JastrowMcMillian(5, 2.85, L)
sampler_bench = HeliumSampler(system, mcmillian_bench, 0.5, L)
sampler_bench.thermalize(10000)
mpiprint(f"AR (bench): {sampler_bench.acceptance_rate}")

optimizer = AdamOptimizer(len(psi.parameters), 0.0001)
optimizer_bench = AdamOptimizer(len(mcmillian_bench.parameters), 0.0001)

iter_per_step = 500
samples_per_iter = 5000
plot_samples = 1_000_000
gamma = 0.00001

steps = 500
t_average = 0
E_training = []
b_training = []
b_bench_training = []
parameters = []
for _ in range(steps):
    t0 = time.time()
    H.optimize_wavefunction(
        psi, sampler, iter_per_step, samples_per_iter, optimizer, gamma, False
    )
    H.optimize_wavefunction(
        mcmillian_bench,
        sampler_bench,
        iter_per_step,
        samples_per_iter,
        optimizer_bench,
        0,
        False,
    )
    t1 = time.time() - t0
    t_average = (t_average * _ + t1) / (_ + 1)
    E = H.local_energy_array(sampler, psi, plot_samples) / P
    E_bench = H.local_energy_array(sampler_bench, mcmillian_bench, plot_samples) / P
    E_training.append([np.mean(E_bench), np.mean(E)])
    b_training.append(psi.parameters[0])
    b_bench_training.append(mcmillian_bench.parameters[0])
    parameters.append(psi.parameters[:100])
    eta = timedelta(seconds=round(t_average * (steps - _)))
    mpiprint(
        f"Step {_+1:5d}/{steps:d} - {1 / t1:5.3f} it/s - ETA {eta} - AR = {sampler.acceptance_rate:.4f} - "
        + f"<E> =  {np.mean(np.asarray(E_training)[:,1]):3.5f} ({np.asarray(E_training)[-1,1]:3.5f}) - "
        + f"<E'> = {np.mean(np.asarray(E_training)[:,0]):3.5f} ({np.asarray(E_training)[-1,0]:3.5f}) - "
        + f"E_sem = {np.std(np.asarray(E_training)[:,1]) / np.sqrt(len(E_training)):3.3f}  - "
        + f"params[0] = {np.mean(b_training):3.5f} ({b_training[-1]:3.5f}) "
        + f"bench[0] = {np.mean(b_bench_training):3.5f} ({b_bench_training[-1]:3.5f})"
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

points = 2 ** 23
t0 = time.time()
H.local_energy_array(sampler, psi, 500)
H.local_energy_array(sampler_bench, mcmillian_bench, 500)
t1 = time.time() - t0

eta = timedelta(seconds=round(t1 / 500 * points))
mpiprint(f"Calculating final energy - ETA {eta}")


stats = [
    compute_statistics_for_series(
        H.local_energy_array(sampler_bench, mcmillian_bench, points) / P,
        method="blocking",
    ),
    compute_statistics_for_series(
        H.local_energy_array(sampler, psi, points) / P, method="blocking"
    ),
]
labels = [r"$\psi_{M}$", r"$\psi_{DNN}$"]
mpiprint(stats, pretty=True)
mpiprint(statistics_to_tex(stats, labels, filename=__file__ + ".table.tex"))

if MPI.COMM_WORLD.rank == 0:
    plot_training(E_training, parameters)
