#pragma once
#include "mpi.h"

namespace mpiutil {

void initialize_mpi(void);
void library_onexit(void);

inline int proc_count(void) {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    return world_size;
}

}
