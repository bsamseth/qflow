#include "hamiltonian.hpp"
#include "harmonicoscillatorhamiltonian.hpp"
#include "interactinghamiltonian.hpp"
#include "rbmharmonicoscillatorhamiltonian.hpp"
#include "rbminteractinghamiltonian.hpp"

#include <Eigen/Dense>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_hamiltonian(py::module& main)
{
    auto m  = main.def_submodule("hamiltonians");
    m.doc() = R"doc(
        Hamiltonians
        -----------------------
    )doc";

    py::class_<Hamiltonian>(m, "Hamiltonian")
        .def("external_potential", &Hamiltonian::external_potential)
        .def("internal_potential", &Hamiltonian::internal_potential)
        .def("local_energy",
             py::overload_cast<System&, Wavefunction&>(&Hamiltonian::local_energy,
                                                       py::const_))
        .def("local_energy",
             py::overload_cast<Sampler&, Wavefunction&, long>(
                 &Hamiltonian::local_energy, py::const_))
        .def("local_energy_numeric", &Hamiltonian::local_energy_numeric)
        .def("kinetic_energy", &Hamiltonian::kinetic_energy)
        .def("kinetic_energy_numeric", &Hamiltonian::kinetic_energy_numeric)
        .def("local_energy_gradient", &Hamiltonian::local_energy_gradient)
        .def("optimize_wavefunction", &Hamiltonian::optimize_wavefunction)
        .def("mean_distance", &Hamiltonian::mean_distance)
        .def("onebodydensity", &Hamiltonian::onebodydensity)
        .def("gross_pitaevskii_energy", &Hamiltonian::gross_pitaevskii_energy);

    py::class_<HarmonicOscillatorHamiltonian, Hamiltonian>(
        m, "HarmonicOscillatorHamiltonian")
        .def(py::init<Real, Real, Real>(),
             py::arg("omega_z") = 1,
             py::arg("a")       = 0,
             py::arg("h")       = 0.001);

    py::class_<InteractingHamiltonian, HarmonicOscillatorHamiltonian>(
        m, "InteractingHamiltonian")
        .def(py::init<Real, Real, Real>(),
             py::arg("omega_z") = 1,
             py::arg("a")       = 0,
             py::arg("h")       = 0.001);

    py::class_<RBMHarmonicOscillatorHamiltonian, Hamiltonian>(
        m, "RBMHarmonicOscillatorHamiltonian")
        .def(py::init<Real>(), py::arg("omega") = 1);

    py::class_<RBMInteractingHamiltonian, RBMHarmonicOscillatorHamiltonian>(
        m, "RBMInteractingHamiltonian")
        .def(py::init<Real>(), py::arg("omega") = 1);
}
