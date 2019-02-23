from typing import Optional, Sequence
import numpy as np
import matplotlib.pyplot as plt
from .hamiltonians import Hamiltonian
from .wavefunctions import Wavefunction
from .samplers import Sampler
from .optimizers import SgdOptimizer


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
