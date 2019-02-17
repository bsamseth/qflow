#include <cstdlib>
#include "mpiutil.hpp"

namespace mpiutil {

void initialize_mpi() {
    int inited;

    MPI_Initialized(&inited);
    if (!inited)
    {
        MPI_Init(nullptr, nullptr);
        atexit(library_onexit);
    }
}

void library_onexit() {
    int finalized;

    MPI_Finalized(&finalized);
    if (!finalized)
        MPI_Finalize();
}

}
