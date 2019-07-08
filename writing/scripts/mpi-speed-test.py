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
ops = np.array([4.99788232e+06, 9.86371553e+06, 2.39456677e+07, 3.31351428e+07,
       3.76873436e+07, 4.15211342e+07, 4.62048929e+07, 5.04229142e+07,
       5.42592367e+07, 5.80505597e+07, 6.11052358e+07, 6.72661224e+07,
       6.91140745e+07, 7.28165158e+07, 7.86169683e+07, 8.31032749e+07,
       8.86181430e+07, 9.32485182e+07, 9.65174129e+07, 1.00534758e+08,
       1.09319625e+08, 1.13993393e+08, 1.18265646e+08, 1.23141165e+08,
       1.25659675e+08, 1.32459134e+08, 1.35795199e+08, 1.39389651e+08,
       1.43589562e+08, 1.48588191e+08, 1.52085216e+08, 1.57127691e+08,
       1.60144623e+08, 1.56155885e+08, 1.67013831e+08, 1.75196745e+08,
       1.77796028e+08, 1.83176171e+08, 1.86734615e+08, 1.90010606e+08,
       1.98068571e+08, 2.02769712e+08, 2.08559638e+08, 2.16177875e+08,
       2.18834269e+08, 2.60500653e+08, 3.02547609e+08, 3.46540375e+08,
       4.20669746e+08, 5.36223780e+08, 6.13017360e+08, 7.49667897e+08,
       7.54151378e+08, 8.98106825e+08, 1.03162828e+09, 1.03653405e+09,
       1.22005665e+09, 1.49703791e+09, 1.55704074e+09, 1.47485020e+09,
       1.54811846e+09, 1.63184827e+09, 2.00868552e+09])
cpus = np.array([  1,   3,   4,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,
        17,  18,  19,  20,  21,  22,  23,  24,  25,  27,  28,  29,  30,
        31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,
        44,  45,  46,  48,  49,  50,  60,  70,  80,  90, 100, 150, 175,
       200, 225, 250, 275, 300, 325, 350, 400, 425, 450, 475])


plt.loglog(cpus, ops, label="Actual")
plt.loglog(cpus, ops[0] * cpus, "--", label="Ideal speedup")
plt.legend()
plt.xlabel("Cores")
plt.ylabel("Iterations per second")

import matplotlib2tikz

matplotlib2tikz.save(__file__ + ".tex")
plt.show()
