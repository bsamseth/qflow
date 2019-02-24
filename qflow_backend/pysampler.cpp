#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

#include "sampler.hpp"
#include "metropolissampler.hpp"
#include "importancesampler.hpp"
#include "gibbssampler.hpp"

namespace py = pybind11;

void init_sampler(py::module& main) {
    auto m = main.def_submodule("samplers");
    m.doc() = R"doc(
        Samplers
        -----------------------
    )doc";

    py::class_<Sampler>(m, "Sampler")
        .def("next_configuration", &Sampler::next_configuration)
        .def("thermalize", &Sampler::thermalize)
        .def_property_readonly("acceptance_rate", &Sampler::get_acceptance_rate);


    py::class_<MetropolisSampler, Sampler>(m, "MetropolisSampler")
        .def(py::init<const System&, Wavefunction&, Real>(),
             py::arg("system"), py::arg("wavefunction"), py::arg("step_size") = 1);
    py::class_<ImportanceSampler, Sampler>(m, "ImportanceSampler")
        .def(py::init<const System&, Wavefunction&, Real>(),
             py::arg("system"), py::arg("wavefunction"), py::arg("step_size") = 0.1);
    py::class_<GibbsSampler, Sampler>(m, "GibbsSampler")
        .def(py::init<const System&, RBMWavefunction&>(),
             py::arg("system"), py::arg("rbm_wavefunction"));
}


