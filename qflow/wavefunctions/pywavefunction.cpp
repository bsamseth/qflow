#include "dnn.hpp"
#include "fixedwavefunction.hpp"
#include "hardspherewavefunction.hpp"
#include "inputsorter.hpp"
#include "jastrowmcmillian.hpp"
#include "jastroworion.hpp"
#include "jastrowpade.hpp"
#include "rbmsymmetricwavefunction.hpp"
#include "rbmwavefunction.hpp"
#include "simplegaussian.hpp"
#include "wavefunction.hpp"
#include "wavefunctionpooling.hpp"
#include "wavefunctionproduct.hpp"

#include <Eigen/Dense>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void init_nn(py::module&);

void init_wavefunction(py::module& main)
{
    auto m  = main.def_submodule("wavefunctions");
    m.doc() = R"doc(
Wavefunctions
-------------

The wavefunction module defines a set of pre-built components that can be used to
set up a variety of different trial wavefunctions.

)doc";

    init_nn(m);

    py::class_<Wavefunction>(m, "Wavefunction", R"doc(
The :class:`Wavefunction` class provides a unified abstraction for all wavefunctions.

All other code expecting a wavefunction instance will take a :class:`Wavefunction` reference.

In this context, a trial wavefunction is defied to be a class that can:

- Have zero or more parameters, :math:`\vec\alpha`
- Evaluate given some system configuraiton :math:`\mathbf{X}`:

.. math::
    \text{Evaluation}_\Psi(\mathbf{X} = \Psi(\mathbf{X})

- Compute the gradient w.r.t. each variational parameter, :math:`\vec{\alpha}`:

.. math::

    \text{Gradient}_\Psi(\mathbf{X}) = \frac{1}{\Psi(\mathbf{X})}
        \frac{\partial \Psi(\mathbf{X})}{\partial\vec\alpha}

- Compute the drift force w.r.t. particle :math:`k`'s dimensions ional coordinate :math:`l`:

.. math::

    \text{Drift}_{\Psi(\mathbf{X}), k,l} = \frac{2}{\Psi(\mathbf{X})}
        \frac{\partial \Psi(\mathbf{X})}{\partial X_{kl}}

- Compute the laplacian of a system of :math:`N` particles in :math:`D` dimensions:

.. math::

    \text{Laplacian}_\Psi(\mathbf{X}) = \sum_{k=1}^N\sum_{l=1}^D
        \frac{1}{\Psi(\mathbf{X})}\frac{\partial^2 \Psi(\mathbf{X})}{\partial X_{kl}^2}

.. note::
    Wavefunctions are limited to real-valued functions only - there is no
    support for complex valued wavefunctions at this time.

)doc")

        .def("__call__", &Wavefunction::operator(), py::arg("system"), R"doc(
Return the evaluation of the wavefunction for the given system

.. math::

    \text{Evaluation}_\Psi(\mathbf{X} = \Psi(\mathbf{X})
)doc")

        .def("gradient", &Wavefunction::gradient, py::arg("system"), R"doc(
Return the gradient w.r.t. each variational parameter, divided by the evaluation:

.. math::
    \text{Gradient}_\Psi(\mathbf{X}) = \frac{1}{\Psi(\mathbf{X})}
        \frac{\partial \Psi(\mathbf{X})}{\partial\vec\alpha}
)doc")

        .def("laplacian", &Wavefunction::laplacian, py::arg("system"), R"doc(

.. math::

    \text{Laplacian}_\Psi(\mathbf{X}) = \sum_{k=1}^N\sum_{l=1}^D
        \frac{1}{\Psi(\mathbf{X})}\frac{\partial^2 \Psi(\mathbf{X})}{\partial X_{kl}^2}
)doc")
        .def("drift_force",
             py::overload_cast<const System&, int, int>(&Wavefunction::drift_force),
             py::arg("system"),
             py::arg("k"),
             py::arg("l"),
             R"doc(

Return the drift force w.r.t. particle :math:`k`'s dimensions ional coordinate :math:`l`:

.. math::

    \text{Drift}_{\Psi(\mathbf{X}), k, l} = \frac{2}{\Psi(\mathbf{X})}
        \frac{\partial \Psi(\mathbf{X})}{\partial X_{kl}}
 )doc")

        .def("drift_force",
             py::overload_cast<const System&>(&Wavefunction::drift_force),
             py::arg("system"),
             R"doc(
Return a list of drift forces for all particles and all dimensions.

The list will be one-dimensional, such that
:math:`\text{Drift}_{\Psi(\mathbf{X}), k,l}` is at index ``k * D + l``.
 )doc")
        .def("symmetry_metric",
             &Wavefunction::symmetry_metric,
             py::arg("sampler"),
             py::arg("samples"),
             py::arg("max_permutations") = 100,
             R"doc(
Return an estimate of the symmetry of the wavefunction.

This is defined as follows:

.. math::
    S(\Psi) =  \frac{\int_{-\infty}^\infty \text{d}\mathbf X\left|\frac{1}{n!}
    \sum_{\mathbf\alpha\in \mathcal{P}_n} P_{\mathbf\alpha}\Psi\right|^2}{\int_{-\infty}^\infty\text{d}\mathbf X
    \max_{\mathbf\alpha\in \mathcal{P}} \left|P_{\mathbf\alpha}\Psi\right|^2}

Here :math:`P_\mathbf{\alpha}\Psi` denotes applying the permutation
:math:`\mathbf{\alpha}` to the input system before evaluating the wavefunction,
and :math:`\mathcal{P}_n` is the set of all permutations of :math:`n` particles.

Properties of this metric:

    - It takes values in :math:`[0, 1]`
    - For symmetric wavefunctions, it equals exactly 1
    - For anti-symmetric wavefunctions, it equals exactly 0
    - Any other function evaluates in :math:`(0, 1)`

This integral will be approximated with Monte Carlo integration using samples from the provided
sampling strategy, and using the specified number of samples. While the integral is defied over all
permutations, a maximum of ``max_permutations`` will be used to make it tractable for large ``N``.
 )doc")
        .def_property(
            "parameters",
            py::overload_cast<>(&Wavefunction::get_parameters, py::const_),
            py::overload_cast<const RowVector&>(&Wavefunction::set_parameters),
            R"doc(
List of all variational parameters. Supports both read and write.
)doc");

    py::class_<FixedWavefunction, Wavefunction>(m, "FixedWavefunction", R"doc(
Wrapper for any wavefunction that fixes the variational parameters.

This means that if this wavefunction is used in a optimization call, its
parameters will not change. This is useful if parts of a wavefunction should be
held constant, while another is allowed to be optimized.
)doc")
        .def(py::init<Wavefunction&>(), py::arg("wavefunction"));

    py::class_<WavefunctionProduct, Wavefunction>(m, "WavefunctionProduct", R"doc(
Defines a wavefunction that acts like the product of two others.

All derivatives will be suitably derived, so this is a simple way to produce
compound expressions.
)doc")
        .def(py::init<Wavefunction&, Wavefunction&>(),
             py::arg("Psi_1"),
             py::arg("Psi_2"));

    py::class_<SumPooling, Wavefunction>(m, "SumPooling", R"doc(
This wavefunction expects a wavefunction of two particles,
:math:`f(\mathbf{x}_1, \mathbf{x}_2)`, and represents the following compound
expression:

.. math::
    \Psi(\mathbf{X}) = \sum_{i \neq j}^N f(\mathbf{X}_i, \mathbf{X}_j)

This is guaranteed to produce a permutation symmetric wavefunction, given any suitable
inner wavefunction :math:`f`.
)doc")
        .def(py::init<Wavefunction&>());

    py::class_<InputSorter, Wavefunction>(m, "InputSorter")
        .def(py::init<Wavefunction&>());

    py::class_<SimpleGaussian, Wavefunction>(m, "SimpleGaussian", R"doc(
A product of gaussians:

.. math::

    \Psi(\mathbf{X}) = \prod_{i=1}^N e^{-\alpha ||\mathbf{X}_i||^2}

The only variational parameter is :math:`\alpha`.

For 3D systems, an optional *fixed* parameter :math:`\beta` can be specified, which changes the above definition to:

.. math::

    \Psi(\mathbf{X}) = \prod_{i=1}^N e^{-\alpha\left(X_{i,1}^2 + X_{i,2}^2 + \beta X_{i,3}^2\right)}
)doc")
        .def(py::init<Real, Real>(), py::arg("alpha") = 0.5, py::arg("beta") = 1);

    py::class_<HardSphereWavefunction, SimpleGaussian>(
        m, "HardSphereWavefunction", R"doc(
A product of gaussians and pairwise correlation factors:

.. math::

    \Psi(\mathbf{X}) = \prod_{i=1}^N e^{-\alpha\left(X_{i,1}^2 + X_{i,2}^2 + \beta X_{i,3}^2\right)}
        \prod_{j = i + 1}^{N} \begin{cases} 0 &\text{if}\ \  ||\mathbf{X}_i - \mathbf{X}_j|| \leq a\\
                                            1 - \frac{a}{||\mathbf{X}_i - \mathbf{X}_j||} &\text{otherwise}
                              \end{cases}

Similarly to :class:`SimpleGaussian`, the only variational parameter is
:math:`\alpha`, while the other two parameters :math:`\beta` and :math:`a` are
assumed constant.
)doc")
        .def(py::init<Real, Real, Real>(),
             py::arg("alpha") = 0.5,
             py::arg("beta")  = 1,
             py::arg("a")     = 0);

    py::class_<JastrowPade, Wavefunction>(m, "JastrowPade", R"doc(
A correlation term meant to be suitable for particles in a harmonic oscillator potential with
a repulsive Coulomb force causing interactions:

.. math::

    \Psi(\mathbf{X}) = \prod_{i < j}^N e^{\frac{\alpha r_ij}{1 + \beta r_ij}}

)doc")
        .def(py::init<Real, Real, bool>(),
             py::arg("alpha")             = 0.5,
             py::arg("beta")              = 1,
             py::arg("alpha_is_constant") = true);

    py::class_<JastrowOrion, Wavefunction>(m, "JastrowOrion", R"doc(
A correlation term meant to be suitable for particles in a harmonic oscillator potential with
a repulsive Coulomb force causing interactions:

.. math::

    \Psi(\mathbf{X}) = \prod_{i < j}^N
        e^{-\frac{\beta^2}{2}||\mathbf{X_i}-\mathbf{X_j}||^2 +
            |\beta\gamma|||\mathbf{X_i}-\mathbf{X_j}||}


)doc")
        .def(py::init<Real, Real>(), py::arg("beta") = 1.0, py::arg("gamma") = 0);

    py::class_<JastrowMcMillian, Wavefunction>(m, "JastrowMcMillian")
        .def(py::init<int, Real, Real>(), py::arg("n"), py::arg("beta"), py::arg("L"));

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
