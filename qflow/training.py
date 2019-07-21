from typing import Sequence, Callable
from abc import ABC, abstractmethod

from .hamiltonians import Hamiltonian
from .wavefunctions import Wavefunction
from .samplers import Sampler
from .optimizers import SgdOptimizer
from .mpi import mpiprint


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
            mpiprint(
                f"EnergyCallback(iter={iter_count}): {self.__getitem__(-1)}", flush=True
            )


class SymmetryCallback(Callback):
    def __init__(self, samples, permutations=50, verbose=False):
        self.samples, self.permutations, self.verbose = samples, permutations, verbose

    def __call__(self, iter_count, psi, H, sampler, optimizer):
        self.append(psi.symmetry_metric(sampler, self.samples, self.permutations))
        if self.verbose:
            mpiprint(
                f"SymmetryCallback(iter={iter_count}): {self.__getitem__(-1)}",
                flush=True,
            )


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
