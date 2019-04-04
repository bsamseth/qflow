from _qflow_backend import *
from _qflow_backend import _init_mpi, _start_distance_tracking, _stop_distance_tracking, _set_simulation_box_size, _disable_simulation_box, _get_simulation_box_size

__all__ = [
    "wavefunctions",
    "hamiltonians",
    "samplers",
    "optimizers",
    "statistics",
    "training",
]

_init_mpi()

class SimulationBox(object):
    def __init__(self, L):
        self.L = L

    def __enter__(self):
        self.old_L = _get_simulation_box_size()
        _set_simulation_box_size(self.L)

    def __exit__(self, exc_type, exc_value, traceback):
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
