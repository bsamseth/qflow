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
    _, (eax, pax) = plt.subplots(ncols=2)
    eax.plot(energies, label=r"$\langle E_L\rangle$ [a.u]")
    eax.set_xlabel(r"% of training")
    eax.set_ylabel(r"Ground state energy (a.u.)")

    pax.plot(np.asarray(parameters))
    pax.set_xlabel(r"% of training")
    pax.legend([r"$\beta$"])

    matplotlib2tikz.save(__file__ + ".tex")


rho = 0.365 / (2.556) ** 3  # Å^-3
P, D = 32, 3  # Particles, dimensions
L = (P / rho) ** (1 / 3)
system = np.empty((P, D))

H = LennardJones(L)
psi = JastrowMcMillian(5, 2.85, L)

sampler = HeliumSampler(system, psi, 0.5, L)
sampler.thermalize(20000)
mpiprint("Acceptance rate after thermalization:", sampler.acceptance_rate)


psi_energies = EnergyCallback(samples=5_000_000, verbose=True)
psi_parameters = ParameterCallback()

train(
    psi,
    H,
    sampler,
    iters=8000,
    samples=5000,
    gamma=0,
    optimizer=AdamOptimizer(len(psi.parameters), 0.0001),
    call_backs=(psi_energies, psi_parameters),
)

mpiprint("Training complete")
mpiprint(psi.parameters)

if master_rank():
    plot_training(np.asarray(psi_energies) / P, psi_parameters)

stats, labels = [], []

for P, step in zip([32, 64, 256], [0.5, 0.6, 0.8]):
    L = (P / rho) ** (1 / 3)
    system = np.empty((P, D))
    H = LennardJones(L)
    psi_ = JastrowMcMillian(5, 2.95, L)
    psi_.parameters = psi.parameters
    _ = HeliumSampler(system, psi_, step, L)
    samp = HeliumSampler(system, psi_, step, L)
    samp.thermalize(10000)
    mpiprint(f"{P}: AR = {samp.acceptance_rate}")

    stats.append(
        compute_statistics_for_series(
            H.local_energy_array(samp, psi_, 2 ** 23) / P, method="blocking"
        )
    )
    labels.append(r"$\psi_M^{(%d)}$" % P)
    mpiprint(f"{P}:", end="")
    mpiprint(stats[-1], pretty=True)


mpiprint(statistics_to_tex(stats, labels, filename=__file__ + ".table.tex"))
