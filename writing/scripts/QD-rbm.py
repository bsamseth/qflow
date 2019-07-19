import numpy as np
import matplotlib.pyplot as plt
import matplotlib2tikz

from qflow.wavefunctions import RBMWavefunction, InputSorter
from qflow.hamiltonians import CoulombHarmonicOscillator
from qflow.samplers import ImportanceSampler
from qflow.optimizers import AdamOptimizer
from qflow.training import train, EnergyCallback, SymmetryCallback, ParameterCallback
from qflow.statistics import compute_statistics_for_series, statistics_to_tex
from qflow.mpi import mpiprint, master_rank


def plot_training(energies, symmetries, parameters):
    fig, (eax, pax) = plt.subplots(ncols=2)
    eax.plot(energies, label=r"$\langle E_L\rangle$")
    eax.set_ylabel(r"Ground state energy (a.u.)")
    eax.set_xlabel(r"% of training")
    eax.axhline(y=3, label="Exact", linestyle="--", color="k", alpha=0.5)
    eax.legend()

    pax.plot(np.asarray(parameters))
    pax.set_xlabel(r"% of training")

    matplotlib2tikz.save(__file__ + ".tex")

    fig, (sax, wax) = plt.subplots(ncols=2)
    sax.semilogx(symmetries, label=r"$S(\psi_{RBM})$")
    sax.set_ylabel("Symmetry")
    sax.set_xlabel(r"% of training")
    sax.legend(loc="lower right")

    w = np.asarray(parameters[-1])[P * D + N :].reshape(P * D, N)
    wax.matshow(w)
    wax.set_xlabel(r"$\mathbf{W}$")

    matplotlib2tikz.save(__file__ + ".symmetry.tex")


P, D, N = 2, 2, 4  # Particles, dimensions, hidden nodes.
system = np.empty((P, D))
H = CoulombHarmonicOscillator()
psi = RBMWavefunction(P * D, N)
psi_sorted = InputSorter(psi)
psi_sampler = ImportanceSampler(system, psi, step_size=0.1)
psi_sorted_sampler = ImportanceSampler(system, psi_sorted, step_size=0.1)

psi_energies = EnergyCallback(samples=1_000_000, verbose=True)
psi_symmetries = SymmetryCallback(samples=1_000_000)
psi_parameters = ParameterCallback()

train(
    psi,
    H,
    psi_sampler,
    iters=40000,
    samples=1000,
    gamma=0.001,
    optimizer=AdamOptimizer(len(psi.parameters), 0.005),
    call_backs=(psi_energies, psi_symmetries, psi_parameters),
)

mpiprint("Training complete")


psi_sorted_sampler.thermalize(100_000)
mpiprint(f"Sorted sampler acceptance rate: {psi_sorted_sampler.acceptance_rate}")

stats = [
    compute_statistics_for_series(
        H.local_energy_array(psi_sampler, psi, 2 ** 23), method="blocking"
    ),
    compute_statistics_for_series(
        H.local_energy_array(psi_sorted_sampler, psi_sorted, 2 ** 23), method="blocking"
    ),
]
labels = [r"$\psi_{RBM}$", r"$\psi_{SRBM}$"]

mpiprint(stats, pretty=True)
mpiprint(statistics_to_tex(stats, labels, filename=__file__ + ".table.tex"))

if master_rank():
    plot_training(psi_energies, psi_symmetries, psi_parameters)
    plt.show()
