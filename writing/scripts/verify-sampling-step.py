import numpy as np
import matplotlib.pyplot as plt
import matplotlib2tikz
from tqdm import tqdm

from qflow.wavefunctions import SimpleGaussian
from qflow.hamiltonians import HarmonicOscillator
from qflow.samplers import MetropolisSampler, ImportanceSampler
from qflow.statistics import compute_statistics_for_series
from qflow.mpi import master_rank

N, D = 10, 3
system = np.empty((N, D))
H = HarmonicOscillator(omega_ho=1)
psi = SimpleGaussian(alpha=0.51)

n_thermal = 50000
n_energy = 2 ** 21

arm, ari = [], []
semm, semi = [], []
steps = np.logspace(-6.5, 1.1, 150)
for step in tqdm(steps):
    ms = MetropolisSampler(system, psi, step)
    si = ImportanceSampler(system, psi, step)

    ms.thermalize(n_thermal)
    si.thermalize(n_thermal)

    arm.append(ms.acceptance_rate * 100)
    ari.append(si.acceptance_rate * 100)
    semm.append(
        compute_statistics_for_series(
            H.local_energy_array(ms, psi, n_energy), method="blocking"
        )["sem"]
    )
    semi.append(
        compute_statistics_for_series(
            H.local_energy_array(si, psi, n_energy), method="blocking"
        )["sem"]
    )

if master_rank():
    mask = np.asarray(semi) < np.max(semm)

    fig, ax1 = plt.subplots()
    met, = ax1.semilogx(steps, arm)
    imp, = ax1.semilogx(steps, ari)
    ax1.set_xlabel(r"Step size")
    ax1.set_ylabel(r"Acceptance rate [%]")

    ax2 = ax1.twinx()
    met_sem, = ax2.semilogx(steps[mask], np.asarray(semm)[mask], linestyle=":")
    imp_sem, = ax2.semilogx(steps[mask], np.asarray(semi)[mask], linestyle=":")
    ax2.set_xlabel(r"Step size")
    ax2.set_ylabel(r"Standard error of energy [a.u.]", rotation=-90)

    ax1.legend(
        [met, imp, met_sem, imp_sem],
        ["Metropolis AR", "Importance AR", "Metroplis SE", "Importance SE"],
        loc="upper left",
    )

    matplotlib2tikz.save(__file__ + ".tex")
    plt.show()
