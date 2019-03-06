#include "activation.hpp"
#include "layer.hpp"

#include <Eigen/Dense>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_layer(py::module& main)
{
    auto m  = main.def_submodule("layers");
    m.doc() = R"pbdoc(
        Network Layers
        -----------------------
        .. currentmodule:: layer
        .. autosummary::
           :toctree: _generate
           DenseLayer
    )pbdoc";

    py::class_<layer::DenseLayer>(m, "DenseLayer")
        .def(py::init<int, int, const activation::ActivationFunction&>(),
             py::arg("inputs"),
             py::arg("outputs"),
             py::arg("activation") = activation::identity)
        .def(py::init<int, int, const activation::ReluActivation&>(),
             py::arg("inputs"),
             py::arg("outputs"),
             py::arg("activation"))
        .def(py::init<int, int, const activation::TanhActivation&>(),
             py::arg("inputs"),
             py::arg("outputs"),
             py::arg("activation"))
        .def(py::init<int, int, const activation::ActivationFunction&, Real>(),
             py::arg("inputs"),
             py::arg("outputs"),
             py::arg("activation"),
             py::arg("scale_factor"))
        .def("evaluate", &layer::DenseLayer::forward)
        .def_property_readonly(
            "weights", py::overload_cast<>(&layer::DenseLayer::getWeights, py::const_))
        .def_property_readonly(
            "biases", py::overload_cast<>(&layer::DenseLayer::getBiases, py::const_))
        .def_property_readonly("weights_gradient",
                               &layer::DenseLayer::getWeightsGradient)
        .def_property_readonly("bias_gradient", &layer::DenseLayer::getBiasGradient);
}
