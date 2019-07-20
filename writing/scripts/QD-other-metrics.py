import time
from datetime import timedelta
import numpy as np

from qflow.wavefunctions import (
    JastrowPade,
    SimpleGaussian,
    WavefunctionProduct,
    Dnn,
    InputSorter,
)
from qflow.wavefunctions.nn.layers import DenseLayer
from qflow.wavefunctions.nn.activations import tanh, exponential
from qflow.hamiltonians import CoulombHarmonicOscillator
from qflow.samplers import ImportanceSampler
from qflow.statistics import compute_statistics_for_series, statistics_to_tex
from qflow.mpi import mpiprint


P, D = 2, 2  # Particles, dimensions
system = np.empty((P, D))
H = CoulombHarmonicOscillator()

# Wave functions:
simple_gaussian = SimpleGaussian(alpha=0.5)
jastrow = JastrowPade(alpha=1, beta=1)
simple_and_jastrow = WavefunctionProduct(simple_gaussian, jastrow)

layers = [
    DenseLayer(P * D, 32, activation=tanh, scale_factor=0.001),
    DenseLayer(32, 16, activation=tanh),
    DenseLayer(16, 1, activation=exponential),
]
dnn = Dnn()
for l in layers:
    dnn.add_layer(l)
psi = WavefunctionProduct(simple_and_jastrow, dnn)
psi_sampler = ImportanceSampler(system, psi, step_size=0.1)
psi.parameters = np.loadtxt("Dnn-parameters-QD.txt")

# Sorted
simple_gaussian2 = SimpleGaussian(alpha=0.5)
jastrow2 = JastrowPade(alpha=1, beta=1)
simple_and_jastrow2 = WavefunctionProduct(simple_gaussian2, jastrow2)

layers2 = [
    DenseLayer(P * D, 32, activation=tanh, scale_factor=0.001),
    DenseLayer(32, 16, activation=tanh),
    DenseLayer(16, 1, activation=exponential),
]
dnn2 = Dnn()
for l in layers2:
    dnn2.add_layer(l)
psi_sorted_base = WavefunctionProduct(simple_and_jastrow2, dnn2)
psi_sorted = InputSorter(psi_sorted_base)
psi_sorted.parameters = psi.parameters
psi_sorted_sampler = ImportanceSampler(system, psi_sorted, step_size=0.1)


# Benchmark:
simple_gaussian_bench = SimpleGaussian(alpha=0.5)
jastrow_bench = JastrowPade(alpha=1, beta=1)
psi_bench = WavefunctionProduct(simple_gaussian_bench, jastrow_bench)
psi_bench.parameters = [0.494_821_73, 1, 1, 0.397_401_86]
psi_bench_sampler = ImportanceSampler(system, psi_bench, step_size=0.1)


wavefuncs = [psi_bench, psi, psi_sorted]
samplers = [psi_bench_sampler, psi_sampler, psi_sorted_sampler]

for s in samplers:
    s.thermalize(10000)


evaluation_points = 2 ** 24
t0 = time.time()
H.mean_squared_radius_array(psi_sampler, 500)
H.mean_radius_array(psi_sampler, 500)
H.mean_distance_array(psi_sampler, P * 500)
t1 = time.time() - t0
eta = timedelta(seconds=round(t1 / 500 * evaluation_points))
mpiprint(f"Calculating final energy - ETA {eta}")

labels = [r"$\psi_{PJ}$", r"$\psi_{DNN}$", r"$\psi_{SDNN}$", r"$\hat{\psi}_{SDNN}$"]

r2_stats = [
    compute_statistics_for_series(
        H.mean_squared_radius_array(s, evaluation_points), method="blocking"
    )
    for s in samplers
]
mpiprint(
    statistics_to_tex(
        r2_stats,
        labels,
        filename=__file__ + ".r2-table.tex",
        quantity_name="$\\langle r^2\\rangle$",
    )
)
r_stats = [
    compute_statistics_for_series(
        H.mean_radius_array(s, evaluation_points), method="blocking"
    )
    for s in samplers
]
mpiprint(
    statistics_to_tex(
        r_stats,
        labels,
        filename=__file__ + ".r-table.tex",
        quantity_name="$\\langle r\\rangle$",
    )
)
rij_stats = [
    compute_statistics_for_series(
        H.mean_distance_array(s, evaluation_points * P), method="blocking"
    )
    for s in samplers
]
mpiprint(
    statistics_to_tex(
        rij_stats,
        labels,
        filename=__file__ + ".rij-table.tex",
        quantity_name="$\\langle r_{12}\\rangle$",
    )
)
