import numpy as np
import matplotlib.pyplot as plt
import matplotlib2tikz

from qflow.wavefunctions import SimpleGaussian
from qflow.hamiltonians import HarmonicOscillator
from qflow.samplers import ImportanceSampler
from qflow.mpi import master_rank

N, D = 100, 3
system = np.empty((N, D))
H = HarmonicOscillator(omega_ho=1)
psi = SimpleGaussian(alpha=0.5)
sampler = ImportanceSampler(system, psi, 0.1)

sampler.thermalize(10000)
max_r = 2.5
samples = 2 ** 23
n_bins = 100
r = np.linspace(0, max_r, n_bins)
bins = H.onebodydensity(sampler, n_bins, max_r, samples)
rho = bins / np.trapz(bins, x=r)
exact = np.exp(-r ** 2)
exact /= np.trapz(exact, x=r)

if master_rank():
    print(rho)
    plt.plot(r, rho, label=r"$\psi_G$")
    plt.plot(r, exact, "--", label=r"$\exp(-r^2)$")
    plt.legend(loc="upper right")
    plt.xlabel(r"$r_1$")
    plt.ylabel(r"Normalized density $\rho(r_1)$")
    matplotlib2tikz.save(__file__ + ".tex")
    plt.show()
