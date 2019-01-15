#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>

#include "dnn.hpp"

namespace py = pybind11;

void init_dnn(py::module& m) {

    py::class_<Dnn>(m, "Dnn")
        .def(py::init<>())
        .def("add_layer", &Dnn::addLayer)
        .def("evaluate", &Dnn::evaluate)
        .def("parameter_gradient", &Dnn::parameterGradient)
        .def("gradient", &Dnn::gradient)
        .def("laplace", &Dnn::laplace)
        .def_property_readonly("layers", &Dnn::getLayers);
}
