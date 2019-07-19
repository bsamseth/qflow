import numpy as np
import matplotlib.pyplot as plt
import matplotlib2tikz

from qflow.wavefunctions import SimpleGaussian
from qflow.hamiltonians import HarmonicOscillator
from qflow.samplers import ImportanceSampler
from qflow.mpi import master_rank

N, D = 10, 3
system = np.empty((N, D))
H = HarmonicOscillator(omega_ho=1)
psi = SimpleGaussian(alpha=0.5)
sampler = ImportanceSampler(system, psi, 0.1)

sampler.thermalize(10000)
max_r = 1.5
samples = 2 ** 25
n_bins = 20
r = np.linspace(0, max_r, n_bins)
bins = H.twobodydensity(sampler, n_bins, max_r, samples)

rho = bins / np.trapz(np.trapz(bins, x=r), x=r)
rho /= np.max(rho)

rx, ry = np.meshgrid(r, r)
exact = np.exp(-rx ** 2 - ry ** 2)
exact /= np.trapz(np.trapz(exact, x=r), x=r)
exact /= np.max(exact)

if master_rank():
    plt.contourf(rx, ry, rho, vmin=0, vmax=1)
    plt.colorbar()
    plt.contour(rx, ry, exact, linestyles="dashed", linewidths=3, vmin=0, vmax=1)
    plt.xlabel(r"$r_1$")
    plt.ylabel(r"$r_2$")
    plt.title(r"Two-body density $\rho(\mathbf{x}_1, \mathbf{x}_2)$")
    matplotlib2tikz.save(__file__ + ".tex")
    plt.show()
