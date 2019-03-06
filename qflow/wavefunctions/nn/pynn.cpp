#include "activation.hpp"
#include "layer.hpp"

#include <Eigen/Dense>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_activation(py::module&);

void init_layer(py::module&);

void init_nn(py::module& main)
{
    auto m  = main.def_submodule("nn");
    m.doc() = R"pbdoc(
        Neural Networks
        -----------------------
    )pbdoc";

    init_activation(m);
    init_layer(m);
}
