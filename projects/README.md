# Project 1 - VMC on Bosonic Systems

## Build & Run

The project is based on CMake, which means it should be able to build the
project codes on most systems. The only external dependency is Google Tests,
used for the unit tests exclusively. Your system must provide a C++ compiler
with C++14 support.

To build the entire project:
``` bash
> mkdir build && cd build  # Enter a dir where the build files should be placed.
> cmake .. -DCMAKE_BUILD_TYPE=[Debug/Release/Coverage]
> make
```

Running the main script:
``` bash
> ./main.x 

Usage: ./main.x analytic[OFF=0,ON=1] sampler[Metro=0,Imp=1] dim n_bosons n_cycles alpha beta gamma a step n_bins max_radius filename 
```

Running the optimization script:
``` bash
> ./optimize

Usage: ./optimize.x sampler[Metro=0,Imp=1] dim n_bosons alpha_guess beta gamma a step learning_rate n_cycles min_gradient
```

Running the tests:
``` bash
> make gtest
```

Producing test coverage reports (requires `lcov`):
``` bash
> cmake .. -DCMAKE_BUILD_TYPE=Coverage
> make coverage
> open coverage_out/index.html
```

Producing documentation (requires `doxygen`):
``` bash
> make doc
> open html/index.html
```

## Source Structure

```
.
├── CMakeLists.txt
├── Doxyfile.in
├── app
├── cmake
├── data
├── external
├── include
├── python
├── report
├── results
├── src
└── tests
```

### Report

The [PDF report](report/report.pdf), with accompanying source file can be found in [report/](report).


### Source Code

The sources for the project are located in [src/](src/), with the corresponding
header files in [include/](include/).  To get an overview of the codes, it is
recommended to view the header files first. These include the documentation for
all classes and functions.

### Tests

The codes are extensively tested. All the tests are located in [tests/](tests/),
and named according to the files they test.

### Executables

The executables are located in [app/](app/).

### Python

The data processing is done in Python, and the files for this are in
[python/](python/). The notebook [python/analysis.ipynb](python/analysis.ipynb)
shows the production of all the figures and tables used in the report.

__Note__: The Python scripts expect the executable to be in a folder named `build-release` in this folder.
In order to run the Python functions, you will need to name the build directory as such.

### Results

All figures and tables produced are located in this folder.

### Data

Selected data files are stored in [data/](data/). The file format describes the
parameters used to the `main.x` executable.

Note that the files are binary, and not human readable. They can be loaded by
the code in [python/analysis.ipynb](python/analysis.ipynb), or directly using
NumPy:
``` python
numpy.fromfile(filename, count=-1, dtype=numpy.float64)
```
