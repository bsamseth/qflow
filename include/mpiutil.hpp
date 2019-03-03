#pragma once
#include "definitions.hpp"
#include "mpi.h"

#include <type_traits>

namespace mpiutil
{
constexpr auto MPI_REAL_TYPE = sizeof(Real) > sizeof(float) ? MPI_DOUBLE : MPI_FLOAT;

void initialize_mpi(void);
void library_onexit(void);

inline int proc_count(void)
{
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    return world_size;
}

inline int get_rank()
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

}  // namespace mpiutil
