#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

#include "activation.hpp"
#include "layer.hpp"

namespace py = pybind11;

void init_activation(py::module&);

void init_layer(py::module& main) {
    auto m = main.def_submodule("layers");
    m.doc() = R"pbdoc(
        Network Layers
        -----------------------
        .. currentmodule:: layer
        .. autosummary::
           :toctree: _generate
           DenseLayer
    )pbdoc";

    init_activation(m);

    py::class_<layer::DenseLayer>(m, "DenseLayer")
        .def(py::init<int, int, const activation::ActivationFunction&>(),
                py::arg("inputs"), py::arg("outputs"), py::arg("activation") = activation::identity)
        .def(py::init<int, int, const activation::ReluActivation&>(),
                py::arg("inputs"), py::arg("outputs"), py::arg("activation"))
        .def(py::init<int, int, const activation::TanhActivation&>(),
                py::arg("inputs"), py::arg("outputs"), py::arg("activation"))
        .def(py::init<int, int, const activation::ActivationFunction&, Real>(),
                py::arg("inputs"), py::arg("outputs"), py::arg("activation"), py::arg("scale_factor"))
        .def("evaluate", &layer::DenseLayer::forward)
        .def_property_readonly("weights", py::overload_cast<>(&layer::DenseLayer::getWeights, py::const_))
        .def_property_readonly("biases", py::overload_cast<>(&layer::DenseLayer::getBiases, py::const_))
        .def_property_readonly("weights_gradient", &layer::DenseLayer::getWeightsGradient)
        .def_property_readonly("bias_gradient", &layer::DenseLayer::getBiasGradient);


    py::class_<layer::InputLayer, layer::DenseLayer>(m, "InputLayer")
        .def(py::init<int, int, const activation::ActivationFunction&>(),
                py::arg("inputs"), py::arg("outputs"), py::arg("activation") = activation::identity)
        .def(py::init<int, int, const activation::ReluActivation&>(),
                py::arg("inputs"), py::arg("outputs"), py::arg("activation"))
        .def(py::init<int, int, const activation::TanhActivation&>(),
                py::arg("inputs"), py::arg("outputs"), py::arg("activation"))
        .def(py::init<int, int, const activation::ActivationFunction&, Real>(),
                py::arg("inputs"), py::arg("outputs"), py::arg("activation"), py::arg("scale_factor"));

}