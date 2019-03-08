from mpi4py import MPI
from pprint import pprint


def mpiprint(*args, pretty=False, **kwargs):
    if MPI.COMM_WORLD.rank == 0:
        if pretty:
            pprint(*args, *kwargs)
        else:
            print(*args, **kwargs)
