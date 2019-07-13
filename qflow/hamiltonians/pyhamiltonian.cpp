#include "coulombharmonicoscillator.hpp"
#include "hamiltonian.hpp"
#include "hardsphereharmonicoscillator.hpp"
#include "harmonicoscillator.hpp"
#include "hfdhe2.hpp"
#include "lennardjones.hpp"

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
             py::overload_cast<const System&, Wavefunction&>(&Hamiltonian::local_energy,
                                                             py::const_))
        .def("local_energy",
             py::overload_cast<Sampler&, Wavefunction&, long>(
                 &Hamiltonian::local_energy, py::const_))
        .def("local_energy_array", &Hamiltonian::local_energy_array)
        .def("local_energy_numeric", &Hamiltonian::local_energy_numeric)
        .def("local_energy_numeric", &Hamiltonian::local_energy_numeric)
        .def("kinetic_energy", &Hamiltonian::kinetic_energy)
        .def("kinetic_energy_numeric", &Hamiltonian::kinetic_energy_numeric)
        .def("local_energy_gradient", &Hamiltonian::local_energy_gradient)
        .def("optimize_wavefunction", &Hamiltonian::optimize_wavefunction)
        .def("mean_distance_array", &Hamiltonian::mean_distance_array)
        .def("mean_radius_array", &Hamiltonian::mean_radius_array)
        .def("mean_squared_radius_array", &Hamiltonian::mean_squared_radius_array)
        .def("onebodydensity", &Hamiltonian::onebodydensity)
        .def("twobodydensity", &Hamiltonian::twobodydensity);

    py::class_<HarmonicOscillator, Hamiltonian>(m, "HarmonicOscillator")
        .def(py::init<Real, Real, Real>(),
             py::arg("omega_ho") = 1,
             py::arg("omega_z")  = 1,
             py::arg("h")        = NUMMERIC_DIFF_STEP)
        .def("gross_pitaevskii_energy", &HarmonicOscillator::gross_pitaevskii_energy);

    py::class_<HardSphereHarmonicOscillator, HarmonicOscillator>(
        m, "HardSphereHarmonicOscillator")
        .def(py::init<Real, Real, Real, Real>(),
             py::arg("omega_ho") = 1,
             py::arg("omega_z")  = 1,
             py::arg("a")        = 0,
             py::arg("h")        = NUMMERIC_DIFF_STEP);

    py::class_<CoulombHarmonicOscillator, HarmonicOscillator>(
        m, "CoulombHarmonicOscillator")
        .def(py::init<Real, Real, Real>(),
             py::arg("omega_ho") = 1,
             py::arg("omega_z")  = 1,
             py::arg("h")        = NUMMERIC_DIFF_STEP);

    py::class_<LennardJones, Hamiltonian>(m, "LennardJones")
        .def(py::init<Real, Real>(),
             py::arg("box_size"),
             py::arg("h") = NUMMERIC_DIFF_STEP);

    py::class_<HFDHE2, Hamiltonian>(m, "HFDHE2")
        .def(py::init<Real>(), py::arg("h") = NUMMERIC_DIFF_STEP);
}
