import numpy as np
import matplotlib.pyplot as plt
import matplotlib2tikz

from qflow.wavefunctions import SimpleGaussian, RBMWavefunction
from qflow.hamiltonians import HarmonicOscillator
from qflow.samplers import MetropolisSampler, ImportanceSampler
from qflow.optimizers import SgdOptimizer, AdamOptimizer
from qflow.training import train, EnergyCallback, ParameterCallback
from qflow.statistics import compute_statistics_for_series, statistics_to_tex

N, D = 10, 3
system = np.empty((N, D))
H = HarmonicOscillator(omega_ho=1)
psi_G = SimpleGaussian(alpha=0.3)
isampler_G = ImportanceSampler(system, psi_G, 0.1)

isampler_G.thermalize(10000)

E_training = EnergyCallback(samples=50000, verbose=True)
G_training = ParameterCallback()
train(
    psi_G,
    H,
    isampler_G,
    iters=100,
    samples=1000,
    gamma=0,
    optimizer=SgdOptimizer(0.001),
    call_backs=[E_training, G_training],
)
E_training = np.asarray(E_training) / N


fig, (eax, pax) = plt.subplots(ncols=2)
eax.plot(E_training, label=r"$\psi_G$")
eax.plot(
    np.ones_like(E_training) * D / 2,
    label=r"$\Phi$",
    linestyle="--",
    alpha=0.5,
    color="k",
)
eax.set_xlabel(r"% of training")
eax.set_ylabel(r"Ground state energy per particle [a.u.]")
eax.legend()
pax.plot(np.asarray(G_training)[:, 0], label=r"$\alpha$")
pax.set_xlabel(r"% of training")
pax.legend()

matplotlib2tikz.save(__file__ + ".tex")
plt.show()
