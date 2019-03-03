#include "dnn.hpp"
#include "fixedwavefunction.hpp"
#include "hardspherewavefunction.hpp"
#include "jastroworion.hpp"
#include "rbmsymmetricwavefunction.hpp"
#include "rbmwavefunction.hpp"
#include "simplegaussian.hpp"
#include "wavefunction.hpp"
#include "wavefunctionproduct.hpp"

#include <Eigen/Dense>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void init_wavefunction(py::module& main)
{
    auto m  = main.def_submodule("wavefunctions");
    m.doc() = R"doc(
        Wavefunctions
        -----------------------
    )doc";

    py::class_<Wavefunction>(m, "Wavefunction")
        .def("__call__", &Wavefunction::operator())
        .def("gradient", &Wavefunction::gradient)
        .def("laplacian", &Wavefunction::laplacian)
        .def("drift_force",
             py::overload_cast<const System&, int, int>(&Wavefunction::drift_force))
        .def("drift_force",
             py::overload_cast<const System&>(&Wavefunction::drift_force))
        .def("symmetry_metric",
             &Wavefunction::symmetry_metric,
             py::arg("sampler"),
             py::arg("samples"),
             py::arg("max_permutations") = 100)
        .def_property(
            "parameters",
            py::overload_cast<>(&Wavefunction::get_parameters, py::const_),
            py::overload_cast<const RowVector&>(&Wavefunction::set_parameters));

    py::class_<FixedWavefunction, Wavefunction>(m, "FixedWavefunction")
        .def(py::init<Wavefunction&>(), py::arg("wavefunction"));

    py::class_<WavefunctionProduct, Wavefunction>(m, "WavefunctionProduct")
        .def(py::init<Wavefunction&, Wavefunction&>(),
             py::arg("Psi_1"),
             py::arg("Psi_2"));

    py::class_<SimpleGaussian, Wavefunction>(m, "SimpleGaussian")
        .def(py::init<Real, Real>(), py::arg("alpha") = 0.5, py::arg("beta") = 1);

    py::class_<HardSphereWavefunction, SimpleGaussian>(m, "HardSphereWavefunction")
        .def(py::init<Real, Real, Real>(),
             py::arg("alpha") = 0.5,
             py::arg("beta")  = 1,
             py::arg("a")     = 0);

    py::class_<JastrowOrion, Wavefunction>(m, "JastrowOrion")
        .def(py::init<Real, Real>(), py::arg("beta") = 1.0, py::arg("gamma") = 0);

    py::class_<RBMWavefunction, Wavefunction>(m, "RBMWavefunction")
        .def(py::init<int, int, Real, Real>(),
             py::arg("M"),
             py::arg("N"),
             py::arg("sigma2")      = 1,
             py::arg("root_factor") = 1);

    py::class_<RBMSymmetricWavefunction, Wavefunction>(m, "RBMSymmetricWavefunction")
        .def(py::init<int, int, int, Real, Real>(),
             py::arg("M"),
             py::arg("N"),
             py::arg("f"),
             py::arg("sigma2")      = 1,
             py::arg("root_factor") = 1);

    py::class_<Dnn, Wavefunction>(m, "Dnn")
        .def(py::init<>())
        .def("add_layer", &Dnn::addLayer)
        .def_property_readonly("layers", &Dnn::getLayers);
}
