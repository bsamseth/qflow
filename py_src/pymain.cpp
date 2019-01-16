#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_wavefunction(py::module&);
void init_sampler(py::module&);
void init_hamiltonian(py::module&);
void init_activation(py::module&);
void init_layer(py::module&);
void init_dnn(py::module&);

PYBIND11_MODULE(qflow, m) {
    m.doc() = R"pbdoc(
        QFlow - Quantum Variational Monte Carlo Framework
        -----------------------
    )pbdoc";

    init_wavefunction(m);
    init_sampler(m);
    init_hamiltonian(m);
    init_activation(m);
    init_layer(m);
    init_dnn(m);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
