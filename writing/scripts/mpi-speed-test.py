import numpy as np
import matplotlib.pyplot as plt

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
ops = np.array(
    [
        4.99788232e06,
        9.86371553e06,
        2.39456677e07,
        3.31351428e07,
        3.76873436e07,
        4.15211342e07,
        4.62048929e07,
        5.04229142e07,
        5.42592367e07,
        5.80505597e07,
        6.11052358e07,
        6.72661224e07,
        6.91140745e07,
        7.28165158e07,
        7.86169683e07,
        8.31032749e07,
        8.86181430e07,
        9.32485182e07,
        9.65174129e07,
        1.00534758e08,
        1.09319625e08,
        1.13993393e08,
        1.18265646e08,
        1.23141165e08,
        1.25659675e08,
        1.32459134e08,
        1.35795199e08,
        1.39389651e08,
        1.43589562e08,
        1.48588191e08,
        1.52085216e08,
        1.57127691e08,
        1.60144623e08,
        1.56155885e08,
        1.67013831e08,
        1.75196745e08,
        1.77796028e08,
        1.83176171e08,
        1.86734615e08,
        1.90010606e08,
        1.98068571e08,
        2.02769712e08,
        2.08559638e08,
        2.16177875e08,
        2.18834269e08,
        2.60500653e08,
        3.02547609e08,
        3.46540375e08,
        4.20669746e08,
        5.36223780e08,
        6.13017360e08,
        7.49667897e08,
        7.54151378e08,
        8.98106825e08,
        1.03162828e09,
        1.03653405e09,
        1.22005665e09,
        1.49703791e09,
        1.55704074e09,
        1.47485020e09,
        1.54811846e09,
        1.63184827e09,
        2.00868552e09,
    ]
)
cpus = np.array(
    [
        1,
        3,
        4,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        48,
        49,
        50,
        60,
        70,
        80,
        90,
        100,
        150,
        175,
        200,
        225,
        250,
        275,
        300,
        325,
        350,
        400,
        425,
        450,
        475,
    ]
)


plt.loglog(cpus, ops, label="Actual")
plt.loglog(cpus, ops[0] * cpus, "--", label="Ideal speedup")
plt.legend()
plt.xlabel("Cores")
plt.ylabel("Iterations per second")

import matplotlib2tikz

matplotlib2tikz.save(__file__ + ".tex")
plt.show()
