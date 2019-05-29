#include "mpiutil.hpp"

#include "definitions.hpp"

#include <cstdlib>
#include <iostream>

namespace mpiutil
{
void initialize_mpi()
{
    int inited;

    MPI_Initialized(&inited);
    if (!inited)
    {
        MPI_Init(nullptr, nullptr);
        atexit(library_onexit);
    }

    // Seed random number generator differently for each proc.
    std::cout << "seeding rank " << get_rank() << std::endl;
    rand_gen.seed(12345 + get_rank());

}  // namespace mpiutil

void library_onexit()
{
    int finalized;

    MPI_Finalized(&finalized);
    if (!finalized)
        MPI_Finalize();
}

}  // namespace mpiutil
