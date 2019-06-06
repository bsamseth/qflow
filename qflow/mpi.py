from mpi4py import MPI
from pprint import pprint

def master_rank():
    return MPI.COMM_WORLD.rank == 0

def mpiprint(*args, pretty=False, **kwargs):
    if master_rank():
        if pretty:
            pprint(*args, *kwargs)
        else:
            print(*args, **kwargs)
