import faulthandler

faulthandler.enable()
import numpy as np
import matplotlib.pyplot as plt

from qflow.wavefunctions import (
    Dnn,
    SimpleGaussian,
    WavefunctionProduct,
    FixedWavefunction,
    RBMWavefunction,
)
from qflow.layers import DenseLayer
from qflow.layers.activations import sigmoid, tanh, relu, identity, exponential

from qflow.hamiltonians import (
    HarmonicOscillatorHamiltonian,
    RBMHarmonicOscillatorHamiltonian,
    RBMInteractingHamiltonian,
)
from qflow.samplers import MetropolisSampler, ImportanceSampler
from qflow.optimizers import AdamOptimizer


def f(x):
    return np.exp(-0.5 * np.sum(x ** 2, axis=1))


def training_plot(
    target,
    rbm,
    H,
    sampler,
    iters,
    samples,
    gamma=0,
    optimizer=None,
    saveas=None,
    verbose=False,
    plot_points=100,
    return_E=False,
):
    def eval():
        return H.local_energy(sampler.next_configuration(), rbm)

    E = [H.local_energy(sampler, rbm, samples)]
    param = [rbm.parameters[:10]]
    rounds_per_iteration = max(1, iters // plot_points)
    for i in range(min(iters, plot_points)):
        H.optimize_wavefunction(
            rbm, sampler, rounds_per_iteration, samples, optimizer, gamma, False
        )
        E.append(H.local_energy(sampler, rbm, 50000))
        if verbose:
            print(f"{i}/{min(iters, plot_points)}: E_psi = {E[-1]}", flush=True)
        param.append(rbm.parameters[:10])

    param = np.asarray(param)

    if return_E:
        return E

    fig, ax = plt.subplots(sharex=True, nrows=2, ncols=1)
    ax[0].semilogy(np.abs(np.asarray(E) - target), label=r"$\langle E_L\rangle$")
    ax[0].set_ylabel(r"$\langle E_L\rangle$ [a.u.]")
    for i in range(np.shape(param)[1]):
        ax[1].plot(param[:, i], label=r"$\alpha_{%d}$" % i)
    ax[1].set_xlabel("Training iterations (x{})".format(rounds_per_iteration))
    plt.legend()

    if saveas:
        plt.savefig(saveas)


dnn = Dnn()
layers = [
    DenseLayer(4, 64, activation=tanh, scale_factor=0.001),
    DenseLayer(64, 32, activation=tanh),
    DenseLayer(32, 1, activation=exponential),
]
for layer in layers:
    dnn.add_layer(layer)

simple = SimpleGaussian(0.5)
rbm = RBMWavefunction(4, 2)
psi = WavefunctionProduct(rbm, dnn)

# H = RBMHarmonicOscillatorHamiltonian()
H = RBMInteractingHamiltonian()
samp = ImportanceSampler(np.zeros((2, 2)), psi, 0.5)
adam = AdamOptimizer(len(psi.parameters))

# Thermalize
print("Startup E = ", H.local_energy(samp, psi, 2 ** 14))
print("Starting optimization")

H.optimize_wavefunction(psi, samp, 5000, 1000, adam, 0.0000, True)
# training_plot(3.0,  psi, H, samp, 5000, 1000, optimizer=adam, verbose=True, gamma=0.0000)
print("Final E = ", H.local_energy(samp, psi, 2 ** 20))
print("Model parameters:", repr(psi.parameters))
np.save("model-parameters.npy", psi.parameters)
plt.show()
# x = np.linspace(-8, 8, 500).reshape(-1, 1)
# plt.plot(x, f(x), label='True')
# plt.plot(x, [psi(x_i.reshape(1,1)) for x_i in x.ravel()], label='Dnn')
# plt.legend()
# plt.show()
