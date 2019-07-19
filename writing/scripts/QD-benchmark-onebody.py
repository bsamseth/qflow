import numpy as np
import matplotlib.pyplot as plt
import matplotlib2tikz

from qflow.wavefunctions import JastrowPade, SimpleGaussian, WavefunctionProduct
from qflow.hamiltonians import CoulombHarmonicOscillator
from qflow.samplers import ImportanceSampler
from qflow.mpi import master_rank


P, D = 2, 2  # Particles, dimensions
system = np.empty((P, D))
H = CoulombHarmonicOscillator()
simple_gaussian = SimpleGaussian(alpha=0.4950)
jastrow = JastrowPade(alpha=1, beta=0.3978)
psi = WavefunctionProduct(simple_gaussian, jastrow)
sampler = ImportanceSampler(system, psi, step_size=0.1)
sampler.thermalize(10000)

samples = 2 ** 27
n_bins = 100
max_r = 3.0
r = np.linspace(0, max_r, n_bins)
bins = H.onebodydensity(sampler, n_bins, max_r, samples)
rho = bins / np.trapz(bins, x=r)
exact_ideal = np.exp(-r ** 2)
exact_ideal /= np.trapz(exact_ideal, x=r)

if master_rank():
    print(rho)
    plt.plot(r, rho, label=r"$\psi_{PJ}$")
    plt.plot(r, exact_ideal, "--", label=r"$\exp(-r^2)$")
    plt.legend(loc="upper right")
    plt.xlabel(r"$r_1$")
    plt.ylabel(r"Normalized density $\rho(r_1)$")
    matplotlib2tikz.save(__file__ + ".tex")
    plt.show()
