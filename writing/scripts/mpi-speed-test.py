import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from os import system

# program = """
# from time import time
# from qflow.mpi import mpiprint
# from qflow.hamiltonians import HarmonicOscillator
# from qflow.wavefunctions import SimpleGaussian
# from qflow.samplers import MetropolisSampler
# H=HarmonicOscillator()
# psi=SimpleGaussian()
# s=MetropolisSampler([[0]],psi,1)
# t0 = time()
# for _ in range(20):
#     H.local_energy(s,psi,2**23)
# mpiprint((time() - t0) / 20)
# """

# cpus = np.arange(1, 800 + 1)
# for n in tqdm(cpus):
#     system(
#         f"mpiexec -n {n} --oversubscribe pipenv run python -c '{program}' >> mpi-speed-test-results.dat"
#     )

# Following data produced by running the above script on a login node at abel, discarding results for n > 5 (unstable access to processors)
times = np.array([0.40548885, 0.21678371, 0.15273976, 0.11579297, 0.09907522])
cpus = np.arange(1, len(times) + 1)

plt.plot(cpus, times[0] / times, label="Actual")
plt.plot(cpus, cpus, "--", label="Ideal speedup")
plt.legend()
plt.xlabel("Cores")
plt.ylabel("Iterations per second")

import matplotlib2tikz

matplotlib2tikz.save(__file__ + ".tex")
plt.show()
