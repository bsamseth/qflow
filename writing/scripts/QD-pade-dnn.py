import numpy as np
import matplotlib.pyplot as plt
import matplotlib2tikz

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
from qflow.optimizers import AdamOptimizer
from qflow.training import train, EnergyCallback, SymmetryCallback, ParameterCallback
from qflow.statistics import compute_statistics_for_series, statistics_to_tex
from qflow.mpi import mpiprint, master_rank


def plot_training(energies, parameters, symmetries):
    _, (eax, pax) = plt.subplots(ncols=2)
    eax.semilogy(np.abs(3 - np.asarray(energies[2])), label=r"$\psi_{PJ}$")
    eax.semilogy(np.abs(3 - np.asarray(energies[0])), label=r"$\psi_{DNN}$")
    eax.semilogy(np.abs(3 - np.asarray(energies[1])), label=r"$\psi_{SDNN}$")
    eax.set_xlabel(r"% of training")
    eax.set_ylabel(r"Absolute error in $\langle E_L\rangle$ [a.u]")
    eax.legend()

    pax.plot(np.asarray(parameters[0])[:, 4:50])
    pax.set_xlabel(r"% of training")

    matplotlib2tikz.save(__file__ + ".tex")

    _, sax = plt.subplots()
    sax.semilogx(symmetries, label=r"$S(\psi_{DNN})$")
    sax.set_ylabel("Symmetry")
    sax.set_xlabel(r"% of training")
    sax.legend(loc="lower right")

    matplotlib2tikz.save(__file__ + ".symmetry.tex")


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
psi.parameters = psi_sorted.parameters
psi_sorted_sampler = ImportanceSampler(system, psi_sorted, step_size=0.1)


# Benchmark:
simple_gaussian_bench = SimpleGaussian(alpha=0.5)
jastrow_bench = JastrowPade(alpha=1, beta=1)
psi_bench = WavefunctionProduct(simple_gaussian_bench, jastrow_bench)
psi_bench_sampler = ImportanceSampler(system, psi_bench, step_size=0.1)


plot_samples = 1_000_000
iters = 30000
samples = 1000
gamma = 0.0
evaluation_points = 2 ** 23

psi_energies = EnergyCallback(samples=plot_samples, verbose=True)
psi_symmetries = SymmetryCallback(samples=plot_samples)
psi_parameters = ParameterCallback()

train(
    psi,
    H,
    psi_sampler,
    iters=iters,
    samples=samples,
    gamma=gamma,
    optimizer=AdamOptimizer(len(psi.parameters)),
    call_backs=(psi_energies, psi_symmetries, psi_parameters),
)
mpiprint("Training regular dnn complete")

np.savetxt("QD-parameters-dnn-regular.txt", psi.parameters)

psi_sorted_energies = EnergyCallback(samples=plot_samples, verbose=True)
psi_sorted_parameters = ParameterCallback()

train(
    psi_sorted,
    H,
    psi_sorted_sampler,
    iters=iters,
    samples=samples,
    gamma=gamma,
    optimizer=AdamOptimizer(len(psi_sorted.parameters)),
    call_backs=(psi_sorted_energies, psi_sorted_parameters),
)
mpiprint("Training sorted dnn complete")

np.savetxt("QD-parameters-dnn-sorted.txt", psi_sorted.parameters)

psi_bench_energies = EnergyCallback(samples=plot_samples)

train(
    psi_bench,
    H,
    psi_bench_sampler,
    iters=iters,
    samples=samples,
    gamma=0,
    optimizer=AdamOptimizer(len(psi.parameters)),
    call_backs=(psi_bench_energies,),
)

mpiprint("Bench Training complete")

stats = [
    compute_statistics_for_series(
        H.local_energy_array(psi_bench_sampler, psi_bench, evaluation_points),
        method="blocking",
    ),
    compute_statistics_for_series(
        H.local_energy_array(psi_sampler, psi, evaluation_points), method="blocking"
    ),
    compute_statistics_for_series(
        H.local_energy_array(psi_sorted_sampler, psi_sorted, evaluation_points),
        method="blocking",
    ),
]
old = psi_sorted.parameters
psi_sorted.parameters = psi.parameters
psi_sorted_sampler.thermalize(10000)

stats.append(
    compute_statistics_for_series(
        H.local_energy_array(psi_sorted_sampler, psi_sorted, evaluation_points),
        method="blocking",
    )
)
labels = [r"$\psi_{PJ}$", r"$\psi_{DNN}$", r"$\psi_{SDNN}$", r"$\hat{\psi}_{SDNN}$"]

mpiprint(stats, pretty=True)
mpiprint(statistics_to_tex(stats, labels, filename=__file__ + ".table.tex"))
# mpiprint(psi.parameters)

if master_rank():
    plot_training(
        [psi_energies, psi_sorted_energies, psi_bench_energies],
        [psi_parameters, psi_parameters],
        psi_symmetries,
    )
    plt.show()
