from _qflow_backend import *
from _qflow_backend import _init_mpi, _start_distance_tracking, _stop_distance_tracking

__all__ = [
    "wavefunctions",
    "hamiltonians",
    "samplers",
    "optimizers",
    "statistics",
    "training",
]

_init_mpi()


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
