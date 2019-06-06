import numpy as np
import matplotlib.pyplot as plt

from qflow.wavefunctions import SimpleGaussian, JastrowPade, WavefunctionProduct
from qflow.hamiltonians import CoulombHarmonicOscillator
from qflow.samplers import ImportanceSampler
from qflow.optimizers import SgdOptimizer
from qflow.training import train, EnergyCallback, ParameterCallback
from qflow.statistics import compute_statistics_for_series
from qflow.mpi import mpiprint, master_rank

P, D = 2, 2  # Particles, dimensions

# Define Hamiltonian:
H = CoulombHarmonicOscillator(omega_ho=1)

# Define trial wave function:
gaussian = SimpleGaussian(alpha=0.8)
jastrow = JastrowPade(alpha=1, beta=1)
psi = WavefunctionProduct(gaussian, jastrow)

# Set up sampling strategy:
sampler = ImportanceSampler(np.empty((P, D)), psi, step_size=0.1)

# Train wave function:
training_energies = EnergyCallback(samples=100000)
training_params = ParameterCallback()
train(
    psi,
    H,
    sampler,
    iters=150,     # Optimization steps.
    samples=1000,  # MC cycles per optimization step.
    gamma=0,       # Regularization parameter (disabled here).
    optimizer=SgdOptimizer(0.1),
    call_backs=(training_energies, training_params),
)

# With a trained model, time to evaluate!
energy = H.local_energy_array(sampler, psi, 2 ** 21)
stats = compute_statistics_for_series(energy, method="blocking")
mpiprint(stats, pretty=True)


if master_rank():
    fig, (eax, pax) = plt.subplots(ncols=2, sharex=True)
    eax.plot(training_energies)
    eax.set_title(r"$\langle E_L\rangle$ [a.u]")
    pax.plot(training_params)
    pax.set_title("Parameters")
    pax.legend(["Gaussian alpha", "Jastrow Alpha", "Jastrow Beta"])
    import matplotlib2tikz

    matplotlib2tikz.save(__file__ + "_.tex")
    plt.show()
