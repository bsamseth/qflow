#include "distance.hpp"
#include "mpiutil.hpp"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_wavefunction(py::module&);
void init_sampler(py::module&);
void init_hamiltonian(py::module&);
void init_optimizer(py::module&);
void init_distance(py::module&);

PYBIND11_MODULE(_qflow_backend, m)
{
    m.doc() = R"pbdoc(
QFlow - Quantum Variational Monte Carlo Framework
-------------------------------------------------


)pbdoc";

    m.def("_init_mpi", &mpiutil::initialize_mpi);

    init_distance(m);
    init_wavefunction(m);
    init_sampler(m);
    init_hamiltonian(m);
    init_optimizer(m);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}

void init_distance(py::module& m)
{
    m.def("_start_distance_tracking", &Distance::start_tracking);
    m.def("_stop_distance_tracking", &Distance::stop_tracking);
}
