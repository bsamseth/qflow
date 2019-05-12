import numpy as np
import scipy.stats as st
from qflow.samplers import HeliumSampler
from qflow.hamiltonians import LennardJones
from qflow.wavefunctions import JastrowMcMillian
from tqdm import tqdm


rho = 0.365 / (2.556) ** 3  # Å^-3
P, D = 108, 3  # Particles, dimensions
L = (P / rho) ** (1 / 3)
system = np.empty((P, D))

print(f"P = {P:3d}, D = {D:2d}, rho = {rho:3.8f}, L = {L:3.5f}")

H = LennardJones(L)
psi = JastrowMcMillian(5, 3.0, L)
sampler = HeliumSampler(system, psi, 1, L)

psi(sampler.current_system)
print(f"Kinetic energy after setup:", H.kinetic_energy(sampler.current_system, psi) / P)
print(
    f"Potential energy after setup:", H.internal_potential(sampler.current_system) / P
)


assert H.kinetic_energy(sampler.current_system, psi) / P == 11.511_056_762_038_711
assert H.internal_potential(sampler.current_system) / P == -18.593_182_408_467_946
sampler.thermalize(5000)
print(f"Acceptance rate after thermalization: {sampler.acceptance_rate}")
assert sampler.acceptance_rate == 0.751


E_k = []
E_p = []
E_l = []
for _ in tqdm(range(500)):
    for _ in range(P):
        s = sampler.next_configuration()
    E_k.append(H.kinetic_energy(sampler.next_configuration(), psi) / P)
    E_p.append(H.internal_potential(sampler.next_configuration()) / P)
    E_l.append(H.local_energy(sampler.next_configuration(), psi) / P)

print(
    f"Average kinetic energy: {np.mean(E_k)} (sem: {st.sem(E_k)}, CI: {st.t.interval(0.95, len(E_k) - 1, loc=np.mean(E_k), scale=st.sem(E_k))})"
)
print(
    f"Average potential energy: {np.mean(E_p)} (sem: {st.sem(E_p)}, CI: {st.t.interval(0.95, len(E_p) - 1, loc=np.mean(E_p), scale=st.sem(E_p))})"
)
print(
    f"Average local energy: {np.mean(E_l)} (sem: {st.sem(E_l)}, CI: {st.t.interval(0.95, len(E_l) - 1, loc=np.mean(E_l), scale=st.sem(E_l))})"
)
