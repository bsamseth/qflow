# Project 2 - The Restricted Boltzmann Machine Applied to the Quantum Many Body Problem

See [project 1](project1.md) for general folder structure and build instructions.

## New stuff

Mainly, the new additions in the main source code are the following classes:

- `RBMWavefunction`: The class implementing the RBM
- `RBMHarmonicOscillatorHamiltonian`: The basic Hamiltonian. This is very similar to the `HarmonicOscillatorHamiltonian` from project 1, and is only a slight simplification from that class.
- `RBMInteractingHamiltonian`: The full Hamiltonian implementation. 
- `SgdOptimizer`: Now a separate class resposible for producing updates to a training algorithm.
- `AdamOptimizer`: Extension of the basic SGD, implementing the ADAM optimizer. 

In addition, rewrites of the `Wavefunction` and `Hamiltonian` base classes have also been made, so as to make project 2 code fit a bit better with the existing code base.

## Python bindings

The major addition is the use of [cppyy](https://cppyy.readthedocs.io/en/latest/) for automatic generation of Python bindings to the C++ backend code. This allows the entierty of the user-code to be written in Python, while all the heavy-lifting is done by C++ behind the scenes. On a relatively modern system, this should be installable simply through `pip install cppyy`. Do note that this will take about an hour to install, as the backend is quite a large piece of code. The `cppyy` project is also quite new, and actively developed. As such, some issues might occur with getting it installed properly.

Given that `cppyy` has been installed successfully, the notebook [RBM.ipynb](python/RBM.ipynb) can be loaded and can reproduce all results in the report. 

There is also a semi-configurable script in [app/rbm_main.cpp](app/rbm_main.cpp) which can be used to obtain results. However, no benchmarks are provided for the use of this executable, as everything has been done via the notebook.
