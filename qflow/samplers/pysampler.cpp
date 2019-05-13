#include "boxsampler.hpp"
#include "gibbssampler.hpp"
#include "heliumsampler.hpp"
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
--------

The sampling classes are the foundation of the Monte Carlo machinery that VMC
depends on. They provide a way of sampling system configurations (i.e.
positions of all particles) from a possibly unnormalized probability amplitude
(i.e. wavefunctions), which in turn allows us to efficiently evaluate
multidimensional integrals.
)doc";

    py::class_<Sampler>(m, "Sampler", R"doc(
The :class:`Sampler` class provides a unified abstraction for the generation of
successive system instances drawn from some probability distribution (i.e. a
wavefunction).

All other code expecting a sampler instance will take a :class:`Sampler` reference.

The subclasses of :class:`Sampler` will differ only in how they generate new
samples, according to their respective algorithms. As such they have different
parameters that can be set at initialization.
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
)doc")

        .def_property_readonly("current_system", &Sampler::get_current_system);

    py::class_<MetropolisSampler, Sampler>(m, "MetropolisSampler", R"doc(
Implementation of the standard Metropolis-Hastings algorithm, producing a
Markov Chain of system configurations. [1]_

Each configuration is a _random_ perturbation of its predecessor if accepted, or
a copy if rejected. The perturbation is defined as a :math:`D` dimensional
vector of uniform random numbers in :math:`\frac{1}{2}[-s, s]` for some scale
parameter :math:`s`.


Notes
-----
This implementation only changes the coordinates of one particle at a time. This
means that two successive calls to `next_configuration()` will at most differ
in one row of the output. This can be trusted to always be the case, and
potential optimizations can be made based on this knowledge. For instance, if
caching of relative distances is employed, only the distances corresponding to
the changed particle would have to be recalculated.


References
----------
.. [1] W. K. Hastings; Monte Carlo sampling methods using Markov chains and
       their applications, Biometrika, Volume 57, Issue 1, 1 April 1970, Pages 97–109,
       https://doi.org/10.1093/biomet/57.1.97


Examples
--------

    >>> import numpy as np
    >>> from qflow.samplers import MetropolisSampler
    >>> from qflow.wavefunctions import SimpleGaussian
    >>> psi = SimpleGaussian()
    >>> sampler = MetropolisSampler(np.zeros((2, 3)), psi, step_size=1)

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
             py::arg("step_size") = 1,
             R"doc(
Construct a sampler that uses the standard Metropolis algorithm. The
`step_size` determines how different successive configurations will be, and
should be tuned such that the acceptance rate remains high at all times.

Arguments
---------
 )doc");

    py::class_<ImportanceSampler, Sampler>(m, "ImportanceSampler", R"doc(
Modified version of the Metropolis-Hastings algorithm, which employs a smarter
way of generating new samples. [2]_ The variance of integrals computed using
this algorithm tends to be significantly lower compared to
:class:`MetropolisSampler`, at the expense of higher run-time cost.

.. warning::
    Importance sampling may only be used with wavefunctions that implements the
    :class:`Wavefunction.drift_force` method. If this is not fulfilled, the
    program will halt immediately without any way of catching an exception from
    Python.

Notes
-----
This algorithm differs from Metropolis-Hastings in the perturbations made, with
a corresponding change in the acceptance probability. In our case,

.. math::

    X_k^{(i+1)} = X_k^{(i)} + \sqrt{t}\mathcal{n} + t \frac{1}{\Psi}\nabla_k \Psi,

where :math:`X_k^{(i)}` is the coordinates of particle :math:`k` at time step
:math:`i`, :math:`\mathcal{n}` is a random number drawn from the standard normal
distribution and :math:`t` is the step size parameter used to tune how
different successive samples are.

Similarly to :class:`MetropolisSampler`, only one particle is perturbed at a time.


References
----------
.. [2] Reiher, W. (1966), Hammersley, J. M., D. C. Handscomb: Monte Carlo
       Methods. Methuen & Co., London, and John Wiley & Sons, New York, 1964. VII +
       178 S., Preis: 25 s. Biom. J., 8: 209-209. doi:10.1002/bimj.19660080314

Examples
--------
This is used exactly like :class:`MetropolisSampler`, with the only exception being the
meaning of the `step_size`.

    >>> import numpy as np
    >>> from qflow.samplers import ImportanceSampler
    >>> from qflow.wavefunctions import SimpleGaussian
    >>> psi = SimpleGaussian()
    >>> sampler = ImportanceSampler(np.zeros((2, 3)), psi, step_size=0.1)
    >>> sampler.next_configuration()
    array([[ 0.56938477,  0.25037102, -0.50411809],
           [-0.8038079 ,  0.15799471, -0.06645576]])
    >>> sampler.thermalize(100)
    >>> sampler.acceptance_rate
    0.9603960396039604

)doc")
        .def(py::init<const System&, Wavefunction&, Real>(),
             py::arg("system"),
             py::arg("wavefunction"),
             py::arg("step_size") = 0.1);

    py::class_<GibbsSampler, Sampler>(m, "GibbsSampler")
        .def(py::init<const System&, RBMWavefunction&>(),
             py::arg("system"),
             py::arg("rbm_wavefunction"));

    py::class_<BoxMetropolisSampler, MetropolisSampler>(m, "BoxMetropolisSampler")
        .def(py::init<const System&, Wavefunction&, Real, Real>(),
             py::arg("system"),
             py::arg("wavefunction"),
             py::arg("box_size"),
             py::arg("step_size") = 1)
        .def("initialize_from_system", &BoxMetropolisSampler::initialize_from_system);

    py::class_<BoxImportanceSampler, ImportanceSampler>(m, "BoxImportanceSampler")
        .def(py::init<const System&, Wavefunction&, Real, Real>(),
             py::arg("system"),
             py::arg("wavefunction"),
             py::arg("box_size"),
             py::arg("step_size") = 0.1);

    py::class_<HeliumSampler, Sampler>(m, "HeliumSampler")
        .def(py::init<const System&, Wavefunction&, Real, Real>(),
             py::arg("system"),
             py::arg("wavefunction"),
             py::arg("step_size"),
             py::arg("box_size"));
}
