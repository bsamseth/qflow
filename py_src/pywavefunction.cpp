#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

#include "wavefunction.hpp"
#include "simplegaussian.hpp"
#include "interactingwavefunction.hpp"
#include "rbmwavefunction.hpp"
#include "dnn.hpp"

namespace py = pybind11;

void init_wavefunction(py::module& main) {
    auto m = main.def_submodule("wavefunction");
    m.doc() = R"doc(
        Wavefunctions
        -----------------------
    )doc";

    py::class_<Wavefunction>(m, "Wavefunction")
        .def("__call__", &Wavefunction::operator())
        .def("gradient", &Wavefunction::gradient)
        .def("laplacian", &Wavefunction::laplacian)
        .def_property("parameters",
                py::overload_cast<>(&Wavefunction::get_parameters, py::const_),
                py::overload_cast<const RowVector&>(&Wavefunction::set_parameters));

    py::class_<SimpleGaussian, Wavefunction>(m, "SimpleGaussian")
        .def(py::init<Real, Real>(), py::arg("alpha") = 0.5, py::arg("beta") = 1);

    py::class_<InteractingWavefunction, SimpleGaussian>(m, "InteractingWavefunction")
        .def(py::init<Real, Real, Real>(), py::arg("alpha") = 0.5, py::arg("beta") = 1, py::arg("a") = 0);

    py::class_<RBMWavefunction, Wavefunction>(m, "RBMWavefunction")
        .def(py::init<int, int, Real, Real>(),
                py::arg("M"), py::arg("N"), py::arg("sigma2") = 1, py::arg("root_factor") = 1);

    py::class_<Dnn, Wavefunction>(m, "Dnn")
        .def(py::init<>())
        .def("add_layer", &Dnn::addLayer)
        .def_property_readonly("layers", &Dnn::getLayers);
}


