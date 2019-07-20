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


max_r = 1.5
samples = 2 ** 28
n_bins = 50
r = np.linspace(0, max_r, n_bins)
bins = H.twobodydensity(sampler, n_bins, max_r, samples)
bins2 = bins ** 2


if master_rank():
    rho = bins / np.trapz(np.trapz(bins, x=r), x=r)
    rho /= np.max(rho)
    plt.imshow(
        rho,
        vmin=0,
        vmax=1,
        extent=[0, max_r, 0, max_r],
        origin="lower",
        interpolation="bicubic",
    )
    plt.colorbar()
    plt.xlabel(r"$r_1$")
    plt.ylabel(r"$r_2$")
    plt.title(r"Two-body density $\rho(\mathbf{x}_1, \mathbf{x}_2)$")
    matplotlib2tikz.save(__file__ + ".tex", tex_relative_path_to_data="scripts/")

    plt.figure()

    rho = bins2 / np.trapz(np.trapz(bins2, x=r), x=r)
    rho /= np.max(rho)
    plt.imshow(
        rho,
        vmin=0,
        vmax=1,
        extent=[0, max_r, 0, max_r],
        origin="lower",
        interpolation="bicubic",
    )
    plt.colorbar()
    plt.xlabel(r"$r_1$")
    plt.ylabel(r"$r_2$")
    plt.title(
        r"Squared two-body density $\left|\rho(\mathbf{x}_1, \mathbf{x}_2)\right|^2$"
    )
    matplotlib2tikz.save(
        __file__ + ".squared.tex", tex_relative_path_to_data="scripts/"
    )
    plt.show()
