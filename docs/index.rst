.. QFlow documentation master file, created by
   sphinx-quickstart on Fri Mar  8 15:48:59 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

QFlow - Quantum Variational Monte Carlo Framework
=================================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   wavefunctions
   samplers
   hamiltonians
   optimizers

This package provides convenient functionality to solve quantum mechanical
many-body systems using the Variational Monte Carlo technique. It provides a
modular interface which allows any combination of trial wavefunctions,
Hamiltonians, sampling strategies and optimization algorithm.

``qflow`` provides a kind Python API to an efficient C++ backend. The Python API
is simply a near-zero cost abstraction layer to the underlying machinery, and
performance can be just as good as using the C++ sources directly.

Features
--------

   - Convenient and consistent Python API
   - Optimized C++ backend
   - Support for near perfect parallelization with MPI
   - A generic Feed-Forward Neural Network implementation which allows to
     include arbitrary deep networks as part of trail wavefunctions

The main selling point of ``qflow``, as opposed to the _many_ VMC implementations
that exist, is the support for arbitrary Feed-Forward Neural Networks. To our
knowledge, no other package allows you to introduce such networks into the
trail wavefunctions for real-valued quantum systems.

The recent project |netket|_ is a more mature package,
with far more contributors working on it. They similarly provide facilities to
do VMC with neural networks, although they seem to be dependent on a _graph_ of
possible sites. This seems to discard systems of freely moving particles, where
particles are to be described by their spacial coordinates, rather than a set
of quantum numbers.

.. note::
   This is entirely based on the understanding of the author of this package,
   and might be incorrect. Please consider |netket|_ for
   yourself, as it very well might be a better fit for your application.


.. |netket| replace:: NetKet
.. _netket: https://netket.org



Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
