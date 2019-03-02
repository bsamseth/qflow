#include "activation.hpp"

#include <Eigen/Dense>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_activation(py::module& main)
{
    auto m  = main.def_submodule("activations");
    m.doc() = R"pbdoc(
        Activation Functions
        -----------------------
        .. currentmodule:: activation
        .. autosummary::
           :toctree: _generate
           identity
           relu
           sigmoid
           tanh
    )pbdoc";

    // Register Classes.
    py::class_<activation::ActivationFunction>(m, "ActivationFunction")
        .def("evaluate", &activation::ActivationFunction::evaluate)
        .def("derivative", &activation::ActivationFunction::derivative)
        .def("dbl_derivative", &activation::ActivationFunction::dblDerivative);
    py::class_<activation::IdentityActivation, activation::ActivationFunction>(
        m, "IdentityActivation");
    py::class_<activation::ReluActivation, activation::ActivationFunction>(
        m, "ReluActivation");
    py::class_<activation::SigmoidActivation, activation::ActivationFunction>(
        m, "SigmoidActivation");
    py::class_<activation::TanhActivation, activation::ActivationFunction>(
        m, "TanhActivation");
    py::class_<activation::ExponentialActivation, activation::ActivationFunction>(
        m, "ExponentialActivation");

    // Register object instances.
    m.attr("identity")    = activation::identity;
    m.attr("relu")        = activation::relu;
    m.attr("sigmoid")     = activation::sigmoid;
    m.attr("tanh")        = activation::tanh;
    m.attr("exponential") = activation::exponential;
}
