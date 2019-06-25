import numpy as np
import matplotlib.pyplot as plt
import matplotlib2tikz

from qflow.wavefunctions import (
    JastrowPade,
    JastrowOrion,
    SimpleGaussian,
    WavefunctionProduct,
    FixedWavefunction,
    Dnn,
    SumPooling,
    InputSorter
)
from qflow.wavefunctions.nn.layers import DenseLayer
from qflow.wavefunctions.nn.activations import sigmoid, tanh, relu, identity, exponential
from qflow.hamiltonians import CoulombHarmonicOscillator
from qflow.samplers import ImportanceSampler
from qflow.optimizers import AdamOptimizer
from qflow.training import train, EnergyCallback, SymmetryCallback, ParameterCallback
from qflow.statistics import compute_statistics_for_series, statistics_to_tex
from qflow.mpi import mpiprint, master_rank


def plot_training(energies, parameters, symmetries):
    fig, (eax, pax) = plt.subplots(ncols=2)
    eax.semilogy(np.abs(3 - np.asarray(energies[1])), label=r"$\psi_{PJ}$")
    eax.semilogy(np.abs(3 - np.asarray(energies[0])), label=r"$\psi_{DNN}$")
    eax.set_xlabel(r"% of training")
    eax.set_ylabel(r"Absolute error in $\langle E_L\rangle$ [a.u]")
    eax.legend()

    pax.plot(np.asarray(parameters)[:, 3:50])
    pax.set_xlabel(r"% of training")

    matplotlib2tikz.save(__file__ + ".tex")

    fig, sax = plt.subplots()
    sax.semilogx(symmetries, label=r"$S(\psi_{DNN})$")
    sax.set_ylabel("Symmetry")
    sax.set_xlabel(r"% of training")
    sax.legend(loc="lower right")

    matplotlib2tikz.save(__file__ + ".symmetry.tex")

    fig, axes = plt.subplots(ncols=2, nrows=3)
    i, j = 3, 3 + 4*32
    p = parameters[-1]
    w0, b0 = p[i:j], p[j: j + 32]
    i, j = j + 32, j + 32 + 32 * 16
    w1, b1 = p[i:j], p[j: j + 16]
    i, j = j + 16, j + 16 + 16
    w2, b2 = p[i:j], p[j: j + 1]

    axes[0][0].matshow(np.reshape(w0, (4,  32)))
    axes[1][0].matshow(np.reshape(w1, (32, 16)), aspect="auto")
    axes[2][0].matshow(np.reshape(w2, (16,  1)), aspect="auto")

    axes[0][1].matshow(np.reshape(b0, (1, -1)), aspect="auto")
    axes[1][1].matshow(np.reshape(b1, (1, -1)), aspect="auto")
    axes[2][1].matshow(np.reshape(b2, (1, -1)), aspect="auto")

    axes[0][0].set_ylabel(r"$\mathbf{W}^{(0)}$")
    axes[1][0].set_ylabel(r"$\mathbf{W}^{(1)}$")
    axes[2][0].set_ylabel(r"$\mathbf{W}^{(2)}$")
    axes[2][0].set_xlabel("Layer weights")

    axes[0][1].set_ylabel(r"$\mathbf{b}^{(0)}$")
    axes[1][1].set_ylabel(r"$\mathbf{b}^{(1)}$")
    axes[2][1].set_ylabel(r"$\mathbf{b}^{(2)}$")
    axes[2][1].set_xlabel("Layer biases")

    axes[0][1].yaxis.set_label_position("right")
    axes[1][1].yaxis.set_label_position("right")
    axes[2][1].yaxis.set_label_position("right")
    fig.tight_layout()

    matplotlib2tikz.save(__file__ + ".weights.tex")


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


# Benchmark:
simple_gaussian_bench = SimpleGaussian(alpha=0.5)
jastrow_bench = JastrowPade(alpha=1, beta=1)
psi_bench = WavefunctionProduct(simple_gaussian_bench, jastrow_bench)
psi_bench_sampler = ImportanceSampler(system, psi_bench, step_size=0.1)


psi_energies = EnergyCallback(samples=1000000, verbose=True)
psi_symmetries = SymmetryCallback(samples=1000000)
psi_parameters = ParameterCallback()

train(
    psi,
    H,
    psi_sampler,
    iters=25000,
    samples=1000,
    gamma=0,
    optimizer=AdamOptimizer(len(psi.parameters)),
    call_backs=(psi_energies, psi_symmetries, psi_parameters),
)
mpiprint("Training complete")

psi_bench_energies = EnergyCallback(samples=1000000)

train(
    psi_bench,
    H,
    psi_bench_sampler,
    iters=25000,
    samples=1000,
    gamma=0,
    optimizer=AdamOptimizer(len(psi.parameters)),
    call_backs=(psi_bench_energies,),
)

mpiprint("Bench Training complete")


stats = [
    compute_statistics_for_series(
        H.local_energy_array(psi_bench_sampler, psi_bench, 2 ** 22),
        method="blocking",
    ),
    compute_statistics_for_series(
        H.local_energy_array(psi_sampler, psi, 2 ** 22), method="blocking"
    ),
]
labels = [r"$\psi_{PJ}$", r"$\psi_{DNN}$"]

mpiprint(stats, pretty=True)
mpiprint(statistics_to_tex(stats, labels, filename=__file__ + ".table.tex"))
# mpiprint(psi.parameters)

if master_rank():
    np.savetxt("Dnn-parameters.txt", psi.parameters)
    plot_training([psi_energies, psi_bench_energies], psi_parameters, psi_symmetries)
    plt.show()

