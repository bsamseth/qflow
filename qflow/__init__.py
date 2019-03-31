from _qflow_backend import *
from _qflow_backend import _init_mpi

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
    def __init__(self, system):
        self.system = system

    def __enter__(self):
        _start_distance_tracking(self.system)

    def __exit__(self):
        _stop_distance_tracking()
