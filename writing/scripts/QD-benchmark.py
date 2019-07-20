import numpy as np
import matplotlib.pyplot as plt
import matplotlib2tikz

from qflow.wavefunctions import JastrowPade, SimpleGaussian, WavefunctionProduct
from qflow.hamiltonians import CoulombHarmonicOscillator
from qflow.samplers import ImportanceSampler
from qflow.optimizers import AdamOptimizer
from qflow.training import EnergyCallback, ParameterCallback, train
from qflow.statistics import compute_statistics_for_series, statistics_to_tex
from qflow.mpi import mpiprint, master_rank


def plot_training(energies, parameters):
    _, (eax, pax) = plt.subplots(ncols=2)
    eax.plot(energies, label=r"$\langle E_L\rangle$ [a.u]")
    eax.set_xlabel(r"% of training")
    eax.axhline(y=3, label="Exact", linestyle="--", color="k", alpha=0.5)
    eax.legend()

    pax.plot(np.asarray(parameters)[:, [0, 3]])
    pax.set_xlabel(r"% of training")
    pax.legend([r"$\alpha_G$", r"$\beta_{PJ}$"])

    matplotlib2tikz.save(__file__ + ".tex")


P, D = 2, 2  # Particles, dimensions
system = np.empty((P, D))
H = CoulombHarmonicOscillator()
simple_gaussian = SimpleGaussian(alpha=0.5)
jastrow = JastrowPade(alpha=1, beta=1)
psi = WavefunctionProduct(simple_gaussian, jastrow)
psi_sampler = ImportanceSampler(system, psi, step_size=0.1)
psi_simple_sampler = ImportanceSampler(system, simple_gaussian, step_size=0.1)

psi_energies = EnergyCallback(samples=100000)
psi_parameters = ParameterCallback()

train(
    psi,
    H,
    psi_sampler,
    iters=2000,
    samples=1000,
    gamma=0,
    optimizer=AdamOptimizer(len(psi.parameters)),
    call_backs=(psi_energies, psi_parameters),
)

mpiprint("Training complete")


stats = [
    compute_statistics_for_series(
        H.local_energy_array(psi_simple_sampler, simple_gaussian, 2 ** 22),
        method="blocking",
    ),
    compute_statistics_for_series(
        H.local_energy_array(psi_sampler, psi, 2 ** 22), method="blocking"
    ),
]
labels = [r"$\Phi$", r"$\psi_{PJ}$"]

mpiprint(stats, pretty=True)
mpiprint(statistics_to_tex(stats, labels, filename=__file__ + ".table.tex"))
mpiprint(psi.parameters)

if master_rank():
    plot_training(psi_energies, psi_parameters)
    plt.show()
