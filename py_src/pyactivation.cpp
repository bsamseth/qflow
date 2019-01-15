#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

#include "activation.hpp"

namespace py = pybind11;

void init_activation(py::module& main) {
    auto m = main.def_submodule("activation");
    m.doc() = R"pbdoc(
        Activation Functions
        -----------------------
        .. currentmodule:: activation
        .. autosummary::
           :toctree: _generate
           identity
           relu
           sigmoid
    )pbdoc";

    // Register Classes.
    py::class_<activation::ActivationFunction>(m, "ActivationFunction")
        .def("evaluate", &activation::ActivationFunction::evaluate)
        .def("derivative", &activation::ActivationFunction::derivative)
        .def("dbl_derivative", &activation::ActivationFunction::dblDerivative);
    py::class_<activation::IdentityActivation, activation::ActivationFunction>(m, "IdentityActivation");
    py::class_<activation::ReluActivation, activation::ActivationFunction>(m, "ReluActivation");
    py::class_<activation::SigmoidActivation, activation::ActivationFunction>(m, "SigmoidActivation");

    // Register object instances.
    m.attr("identity") = activation::identity;
    m.attr("relu") = activation::relu;
    m.attr("sigmoid") = activation::sigmoid;
}

