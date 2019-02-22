import faulthandler

faulthandler.enable()
import numpy as np
from pprint import pprint

from qflow.wavefunctions import (
    Dnn,
    SimpleGaussian,
    WavefunctionProduct,
    FixedWavefunction,
    RBMWavefunction,
)
from qflow.layers import DenseLayer
from qflow.layers.activations import sigmoid, tanh, relu, identity, exponential

from qflow.statistics import compute_statistics_for_series

from qflow.hamiltonians import (
    HarmonicOscillatorHamiltonian,
    RBMHarmonicOscillatorHamiltonian,
    RBMInteractingHamiltonian,
)
from qflow.samplers import MetropolisSampler, ImportanceSampler

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

psi.parameters = np.load("model-parameters.npy")

H = RBMInteractingHamiltonian()
samp = ImportanceSampler(np.zeros((2, 2)), psi, 0.5)

for _ in range(10000):
    samp.next_configuration()

s = samp.next_configuration()
print(psi(s), psi(s[::-1, :]))
E = np.array([H.local_energy(samp.next_configuration(), psi) for _ in range(2 ** 13)])

pprint(compute_statistics_for_series(E))
pprint(compute_statistics_for_series(E, method="blocking"))
