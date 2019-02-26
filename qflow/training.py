from typing import Optional, Sequence, Callable
from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
from .hamiltonians import Hamiltonian
from .wavefunctions import Wavefunction
from .samplers import Sampler
from .optimizers import SgdOptimizer


class Callback(ABC, list):
    @abstractmethod
    def __call__(self, iter_count, psi, H, sampler, optimizer):
        pass


class EnergyCallback(Callback):
    def __init__(self, samples, verbose=False):
        self.samples, self.verbose = samples, verbose

    def __call__(self, iter_count, psi, H, sampler, optimizer):
        self.append(H.local_energy(sampler, psi, self.samples))
        if self.verbose:
            print(f"EnergyCallback(iter={iter_count}): {self.__getitem__(-1)}")


class SymmetryCallback(Callback):
    def __init__(self, samples, permutations=50, verbose=False):
        self.samples, self.permutations, self.verbose = samples, permutations, verbose

    def __call__(self, iter_count, psi, H, sampler, optimizer):
        self.append(psi.symmetry_metric(sampler, self.samples, self.permutations))
        if self.verbose:
            print(f"SymmetryCallback(iter={iter_count}): {self.__getitem__(-1)}")


class ParameterCallback(Callback):
    def __call__(self, iter_count, psi, *args):
        self.append([p for p in psi.parameters])


def train(
    psi: Wavefunction,
    H: Hamiltonian,
    sampler: Sampler,
    iters: int,
    samples: int,
    gamma: float,
    optimizer: SgdOptimizer,
    verbose: bool = False,
    call_backs: Sequence[Callable] = tuple(),
    call_back_resolution: int = 100,
):
    """Train a wavefunction according to settings, with call back functions."""
    completed_iterations = 0
    for iteration in range(call_back_resolution):
        for cb in call_backs:
            cb(completed_iterations, psi, H, sampler, optimizer)

        iterations_now = iters // call_back_resolution + int(
            iteration < iters % call_back_resolution
        )
        H.optimize_wavefunction(
            psi, sampler, iterations_now, samples, optimizer, gamma, verbose
        )
        completed_iterations += iterations_now

    for cb in call_backs:
        cb(completed_iterations, psi, H, sampler, optimizer)


def training_plot(
    psi: Wavefunction,
    H: Hamiltonian,
    sampler: Sampler,
    iters: int,
    samples: int,
    gamma: float,
    optimizer: SgdOptimizer,
    target_energy: Optional[float] = None,
    saveas: Optional[str] = None,
    verbose: bool = False,
    plot_points: int = 100,
    return_E: bool = False,
) -> Optional[Sequence[float]]:
    """
    Optimize the a `Wavefunction` to a `Hamiltonian` and save incremental energy
    evaluations that can be used to produce a plot and/or analyse the energy series.

    TODO: Document.
    """

    E = [H.local_energy(sampler, psi, samples)]
    rounds_per_iteration = max(1, iters // plot_points)
    for i in range(min(iters, plot_points)):
        H.optimize_wavefunction(
            psi, sampler, rounds_per_iteration, samples, optimizer, gamma, False
        )
        E.append(H.local_energy(sampler, psi, 50000))
        if verbose:
            print(f"{i}/{min(iters, plot_points)}: E_psi = {E[-1]}", flush=True)

    fig, ax = plt.subplots()

    if target_energy is not None:
        ax.semilogy(
            np.abs(np.asarray(E) - target_energy), label=r"$\langle E_L\rangle$"
        )
        ax.set_ylabel(r"Absolute error for $\langle E_L\rangle$ [a.u.]")
    else:
        ax.semilogy(E, label=r"$\langle E_L\rangle$")
        ax.set_ylabel(r"Local Energy $\langle E_L\rangle$ [a.u.]")
    ax.set_xlabel("Training iterations (x{})".format(rounds_per_iteration))

    plt.legend(loc="best")

    if saveas:
        plt.savefig(saveas)

    if return_E:
        return E
