import numpy as np
import matplotlib.pyplot as plt
import matplotlib2tikz

from qflow.wavefunctions import RBMWavefunction, SimpleGaussian
from qflow.hamiltonians import HarmonicOscillator
from qflow.samplers import ImportanceSampler
from qflow.optimizers import SgdOptimizer, AdamOptimizer
from qflow.training import train, EnergyCallback, ParameterCallback
from qflow.statistics import compute_statistics_for_series, statistics_to_tex
from qflow.mpi import mpiprint, master_rank

N, D = 1, 1
system = np.empty((N, D))
H = HarmonicOscillator(omega_ho=1)
psi = RBMWavefunction(N * D, 2)
# psi = SimpleGaussian(0.3)
org_params = psi.parameters[:]
sampler = ImportanceSampler(system, psi, 0.5)

labels = [
    r"Sgd($\eta=1$)",
    r"Sgd($\eta=0.1$)",
    r"Sgd($\eta=0.05$)",
    r"Adam($\eta=0.1,\beta_1=0.9$)",
    r"Adam($\eta=0.1,\beta_1=0.8$)",
]
optimizers = [
    SgdOptimizer(1),
    SgdOptimizer(0.1),
    SgdOptimizer(0.05),
    AdamOptimizer(len(psi.parameters), 0.1, 0.9),
    AdamOptimizer(len(psi.parameters), 0.1, 0.8),
]
E = []
for opt in optimizers:
    # psi.parameters = org_params
    psi = RBMWavefunction(N * D, 2)
    # psi = SimpleGaussian(0.8)
    sampler = ImportanceSampler(system, psi, 0.1)
    sampler.thermalize(10000)
    E_training = EnergyCallback(samples=1000000, verbose=True)
    train(
        psi,
        H,
        sampler,
        iters=500,
        samples=1000,
        gamma=0.0,
        optimizer=opt,
        call_backs=[E_training],
        call_back_resolution=50,
    )
    E.append(np.asarray(E_training))


if master_rank():
    fig, ax = plt.subplots()
    ax.set_xlabel(r"% of training")
    ax.set_ylabel(r"Energy error [a.u.]")
    for e, label in zip(E, labels):
        ax.semilogy(np.abs(e / N - D / 2), label=label)
    ax.legend()
    matplotlib2tikz.save(__file__ + ".tex", extra_axis_parameters=["compat=newest", "legend pos=outer north east"])
    plt.show()
