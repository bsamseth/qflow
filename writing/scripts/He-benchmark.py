import numpy as np
import matplotlib.pyplot as plt
import matplotlib2tikz

from qflow.wavefunctions import JastrowMcMillian
from qflow.hamiltonians import LennardJones
from qflow.samplers import HeliumSampler
from qflow.optimizers import AdamOptimizer
from qflow.training import train, EnergyCallback, ParameterCallback
from qflow.statistics import compute_statistics_for_series, statistics_to_tex
from qflow.mpi import mpiprint, master_rank


def plot_training(energies, parameters):
    fig, (eax, pax) = plt.subplots(ncols=2)
    eax.plot(energies, label=r"$\langle E_L\rangle$ [a.u]")
    eax.set_xlabel(r"% of training")

    pax.plot(np.asarray(parameters))
    pax.set_xlabel(r"% of training")
    pax.legend([r"$\alpha_G$", r"$\beta_{PJ}$"])

    matplotlib2tikz.save(__file__ + ".tex")


rho = 0.365 / (2.556) ** 3  # Å^-3
P, D = 32, 3  # Particles, dimensions
L = (P / rho) ** (1 / 3)
system = np.empty((P, D))

H = LennardJones(L)
psi = JastrowMcMillian(5, 2.9, L)

sampler = HeliumSampler(system, psi, 0.5, L)
sampler.thermalize(10000)
mpiprint("Acceptance rate after thermalization:", sampler.acceptance_rate)


psi_energies = EnergyCallback(samples=100, verbose=True)
psi_parameters = ParameterCallback()

train(
    psi,
    H,
    sampler,
    iters=2500,
    samples=1000,
    gamma=0,
    optimizer=AdamOptimizer(len(psi.parameters)),
    call_backs=(psi_energies, psi_parameters),
)

mpiprint("Training complete")


stats = [
    compute_statistics_for_series(
        H.local_energy_array(sampler, psi, 2 ** 20) / P, method="blocking"
    ),
]
labels = [r"$\Phi$", r"$\psi_{PJ}$"]

mpiprint(stats, pretty=True)
mpiprint(statistics_to_tex(stats, labels, filename=__file__ + ".table.tex"))
mpiprint(psi.parameters)

psi_energies = np.asarray(psi_energies) / P

if master_rank():
    plot_training(psi_energies, psi_parameters)
    plt.show()
