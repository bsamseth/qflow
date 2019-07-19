import numpy as np

from qflow.wavefunctions import SimpleGaussian
from qflow.hamiltonians import HarmonicOscillator
from qflow.samplers import ImportanceSampler
from qflow.statistics import compute_statistics_for_series, statistics_to_tex
from qflow.mpi import mpiprint

N, D = 100, 3
system = np.empty((N, D))
H = HarmonicOscillator(omega_ho=1)
psi_opt = SimpleGaussian(alpha=0.5)
psi_nopt = SimpleGaussian(alpha=0.51)
sampler_opt = ImportanceSampler(system, psi_opt, 0.1)
sampler_nopt = ImportanceSampler(system, psi_nopt, 0.1)

sampler_opt.thermalize(10000)
sampler_nopt.thermalize(10000)

samples = 2 ** 23

stats = [
    compute_statistics_for_series(H.local_energy_array(sampler_opt, psi_opt, 100) / N),
    compute_statistics_for_series(
        H.local_energy_array(sampler_nopt, psi_nopt, samples) / N, method="blocking"
    ),
]

labels = [r"$\alpha_G = 0.5$", r"$\alpha_G=0.51$"]

mpiprint(statistics_to_tex(stats, labels, filename=__file__ + ".table1.tex"))


stats = [
    compute_statistics_for_series(H.mean_radius_array(sampler_opt, samples)),
    compute_statistics_for_series(H.mean_squared_radius_array(sampler_opt, samples)),
]

labels = [r"$<r>$", r"$<r^2>$"]

mpiprint(statistics_to_tex(stats, labels, filename=__file__ + ".table2.tex"))
