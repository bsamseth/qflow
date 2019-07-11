import numpy as np
import matplotlib.pyplot as plt
import matplotlib2tikz

from qflow.wavefunctions import SimpleGaussian
from qflow.hamiltonians import HarmonicOscillator
from qflow.samplers import MetropolisSampler
from qflow.statistics import compute_statistics_for_series
from qflow.mpi import mpiprint, master_rank

H = HarmonicOscillator()
psi = SimpleGaussian(alpha=0.5)
sampler = MetropolisSampler(np.zeros((1,1)), psi, 0.5)
sampler.thermalize(10000)

d = 27
r = H.mean_radius_array(sampler, 2**d)

x = r[:]

sem = np.empty(d)
for i in range(0, d):
    sem[i] = np.std(x) / np.sqrt(len(x))
    x = 0.5 * (x[0::2] + x[1::2])

auto_block_sem = compute_statistics_for_series(r, method='blocking')['sem']
auto_d = d - np.argmin(np.abs(sem - auto_block_sem))
if master_rank():
    plt.plot(list(range(d, 0, -1)), sem)
    plt.plot([auto_d], [auto_block_sem], 'ro', label='Optimal estimate')
    plt.xlim(d, 1)
    plt.legend(loc='lower right')
    plt.xlabel('Number of samples [$\log_2 N$]')
    plt.ylabel('Standard error of the mean')
    matplotlib2tikz.save(__file__ + ".tex")
    plt.show()
