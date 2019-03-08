#include "gibbssampler.hpp"
#include "importancesampler.hpp"
#include "metropolissampler.hpp"
#include "sampler.hpp"

#include <Eigen/Dense>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void init_sampler(py::module& main)
{
    auto m  = main.def_submodule("samplers");
    m.doc() = R"doc(
        Samplers
        -----------------------
    )doc";

    py::class_<Sampler>(m, "Sampler", R"doc(
The :class:`Sampler` class provides a unified abstraction for the generation of
successive system instances drawn from some probability distribution (i.e. a
wavefunction).

All other code expecting a sampler instance will take a :class:`Sampler` reference.
)doc")
        .def("next_configuration", &Sampler::next_configuration, R"doc(
Return a newly sampled system configuration.

There is no guarantee that this will always differ between successive calls,
but with a sufficient number of calls the distribution of outputs,
:math:`P(\mathbf{X})` should approximate

.. math::

    P(\mathbf{X}) \simeq  |\Psi(\mathbf{X})|^2

)doc")

        .def("thermalize", &Sampler::thermalize, R"doc(
Generate a given number of samples, discarding all.

This is useful to ensure that the sampler has reached a stable point where
samples are representative of the underlying distribution.
)doc")

        .def_property_readonly("acceptance_rate", &Sampler::get_acceptance_rate, R"doc(
Rate at which newly proposed samples where accepted by the algorithm.
)doc");

    py::class_<MetropolisSampler, Sampler>(m, "MetropolisSampler", R"doc(
Implementation of the standard Metropolis algorithm.

Examples
--------

    >>> import numpy as np
    >>> from qflow.samplers import MetropolisSampler
    >>> from qflow.wavefunctions import SimpleGaussian
    >>> psi = SimpleGaussian()
    >>> sampler = MetropolisSampler(np.zeros((2, 3)), psi)

We can now get samples of the desired size on demand:

    >>> sampler.next_configuration()
    array([[-0.61370758,  0.08936709,  0.15872668],
           [ 0.05973557,  0.07445129, -0.29230947]])

Thermalize the sampler (equivalent to running `next_configuration` the given number of times, only faster):

    >>> sampler.thermalize(100)

Inspect the acceptance rate:

    >>> sampler.acceptance_rate
    0.7326732673267327
)doc")
        .def(py::init<const System&, Wavefunction&, Real>(),
             py::arg("system"),
             py::arg("wavefunction"),
             py::arg("step_size") = 1);
    py::class_<ImportanceSampler, Sampler>(m, "ImportanceSampler")
        .def(py::init<const System&, Wavefunction&, Real>(),
             py::arg("system"),
             py::arg("wavefunction"),
             py::arg("step_size") = 0.1);
    py::class_<GibbsSampler, Sampler>(m, "GibbsSampler")
        .def(py::init<const System&, RBMWavefunction&>(),
             py::arg("system"),
             py::arg("rbm_wavefunction"));
}
