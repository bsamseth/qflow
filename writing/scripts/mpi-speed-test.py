import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from os import system
from multiprocessing import cpu_count

program = """
from time import time
from qflow.mpi import mpiprint
from qflow.hamiltonians import HarmonicOscillator
from qflow.wavefunctions import SimpleGaussian
from qflow.samplers import MetropolisSampler
H=HarmonicOscillator()
psi=SimpleGaussian()
s=MetropolisSampler([[0]],psi,1)
t0 = time()
for _ in range(20):
    H.local_energy(s,psi,1000000)
mpiprint((time() - t0) / 10)
"""

cpus = np.arange(1, cpu_count() + 1)
for n in tqdm(cpus):
    system(
        f"mpiexec -n {n} --oversubscribe pipenv run python -c '{program}' >> mpi-speed-test-results.dat"
    )

times = np.loadtxt("mpi-speed-test-results.dat")
print(times)
system("rm mpi-speed-test-results.dat")

plt.plot(cpus, 1_000_000 / times, label="Actual ops/sec")
plt.plot(cpus, 1_000_000 / times[0] * cpus, "--", label="Linear speedup")
plt.legend()
import matplotlib2tikz

matplotlib2tikz.save(__file__ + ".tex")
plt.show()
