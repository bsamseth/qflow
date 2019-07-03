import numpy as np
from qflow.hamiltonians import HarmonicOscillator
from qflow.wavefunctions import SimpleGaussian
from qflow.samplers import ImportanceSampler
from qflow.statistics import compute_statistics_for_series


def test_ideal_harmonic_oscillator():
    omega, N, D = 1, 5, 3
    H = HarmonicOscillator(omega_ho=omega)
    psi = SimpleGaussian(alpha=0.5 * omega)
    sampler = ImportanceSampler(np.empty((N, D)), psi, 0.1)
    sampler.thermalize(10000)

    # Energy:
    E = compute_statistics_for_series(H.local_energy_array(sampler, psi, 2 ** 10))
    assert N * D * omega / 2 == E["mean"], "Energy should be exactly equal."
    assert E["var"] < 1e-16

    # Squared radius:
    r2 = compute_statistics_for_series(
        H.mean_squared_radius_array(sampler, 2 ** 16), method="blocking"
    )
    print(r2)
    assert (
        abs(3 / omega / 2 - r2["mean"]) < 1.96 * r2["sem"]
    ), "Should be withing 95% CI of analytic result."

    # Radius:
    r = compute_statistics_for_series(
        H.mean_radius_array(sampler, 2 ** 16), method="blocking"
    )
    print(r)
    assert (
        abs(2 / np.sqrt(np.pi * omega) - r["mean"]) < 1.96 * r["sem"]
    ), "Should be withing 95% CI of analytic result."
