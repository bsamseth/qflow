#include "optimizer.hpp"

#include <Eigen/Dense>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_optimizer(py::module& main)
{
    auto m  = main.def_submodule("optimizers");
    m.doc() = R"doc(
        Optimizers
        -----------------------
    )doc";

    py::class_<SgdOptimizer>(m, "SgdOptimizer")
        .def(py::init<Real>(), py::arg("learning_rate") = 0.1);

    py::class_<AdamOptimizer, SgdOptimizer>(m, "AdamOptimizer")
        .def(py::init<std::size_t, Real, Real, Real, Real>(),
             py::arg("n_parameters"),
             py::arg("alpha")   = 0.001,
             py::arg("beta1")   = 0.9,
             py::arg("beta2")   = 0.999,
             py::arg("epsilon") = 1e-8);
}
