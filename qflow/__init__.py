import mpi4py  # Explicitly load mpi4py, which initializes the MPI environment properly.
from _qflow_backend import *
from _qflow_backend import (
    _init_mpi,
    _start_distance_tracking,
    _stop_distance_tracking,
    _set_simulation_box_size,
    _disable_simulation_box,
    _get_simulation_box_size,
)

__all__ = [
    "wavefunctions",
    "hamiltonians",
    "samplers",
    "optimizers",
    "statistics",
    "training",
]

_init_mpi()  # Initializing from Python is not necessary for MPI it self,
             # but rather to seed the random number generators differently
             # for each process.


class SimulationBox(object):
    def __init__(self, L):
        self.L = L

    def __enter__(self):
        self.old_L = _get_simulation_box_size()
        _set_simulation_box_size(self.L)

    def __exit__(self, exc_type, exc_value, traceback):
        if self.old_L < 0:
            _disable_simulation_box()
        else:
            _set_simulation_box_size(self.old_L)


class DistanceCache(object):
    def __init__(self, system, pbc_size=None):
        self.system = system
        self.pbc_size = pbc_size

    def __enter__(self):
        if self.pbc_size is not None:
            _start_distance_tracking(self.system, self.pbc_size)
        else:
            _start_distance_tracking(self.system)

    def __exit__(self, exc_type, exc_value, traceback):
        _stop_distance_tracking()
