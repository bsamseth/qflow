[![Build Status](https://travis-ci.org/bsamseth/cpp-project.svg?branch=master)](https://travis-ci.org/bsamseth/cpp-project)
[![Coverage Status](https://coveralls.io/repos/github/bsamseth/cpp-project/badge.svg?branch=master)](https://coveralls.io/github/bsamseth/cpp-project?branch=master)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/bsamseth/cpp-project/blob/master/LICENCE)
[![Project Status: Inactive - The project has reached a stable, usable state but is no longer being actively developed; support/maintenance will be provided as time allows.](http://www.repostatus.org/badges/latest/inactive.svg)](http://www.repostatus.org/#inactive)

# Boiler plate for C++ projects 

This is a boiler plate for C++ projects. What you get:

- Sources, headers and mains separated in distinct folders
- Access to [Google Tests](https://github.com/google/googletest)
- Use of [CMake](https://cmake.org/) for much easier compiling
- Code documentation with [Doxygen](http://www.stack.nl/~dimitri/doxygen/)
- Continuous testing with [Travis-CI](https://travis-ci.org/)
- Code coverage with [Coveralls.io](https://coveralls.io/)

## Structure
```
.
├── CMakeLists.txt
├── app
│   └── main.cpp
├── include
│   ├── example.h
│   └── exampleConfig.h.in
├── src
│   └── example.cpp
└── tests
    ├── dummy.cpp
    └── main.cpp
```

Sources go in [src/](src/), header files in [include/](include/), main programs in [app/](app), and
tests go in [tests/](tests/) (compiled to `unit_tests.x` by default). 

If you add a new executable, say `app/hello.cpp`, you only need to add the following three lines to [CMakeLists.txt](CMakeLists.txt): 

``` cmake
add_executable(hello.x app/hello.cpp)   # Name of exec. and location of file.
add_dependencies(hello.x engine)        # engine is the library built from src/*.cpp
target_link_libraries(hello.x engine)   # Link the executable to the 'engine'.
```

You can find the example that builds the example in [app/main.cpp](app/main.cpp) under the `Build` section in [CMakeLists.txt](CMakeLists.txt). 
If the executable you made does not use the library in [src/](src), then only the first line is needed.



## Building

Build by making a build directory (i.e. `build/`), run `cmake` in that dir, and then use `make` to build the desired target.

Example:

``` bash
$ mkdir build && cd build
$ cmake .. # argument is location of CMakelists.txt
$ make
$ make gtest
$ ./main.x
```

## .gitignore

The [.gitignore](.gitignore) file is a copy of the [Github C++.gitignore file](https://github.com/github/gitignore/blob/master/C%2B%2B.gitignore),
with the addition of ignoring the build directory (`build/`).


## Services

If repository is activated with Travis-CI, then unit tests will be built and executed on each commit.

If repository is activated with Coveralls, then deployment to Travis will also calculate code coverage and
upload this to Coveralls.io. 

## Setup
When starting a new project, you probably don't want the history of this repository. To start fresh, with just the files
and no history, you simply delete the `.git/` directory and start a new one:

``` bash
$ git clone <link to this repo>
$ cd cpp-project
$ rm -rf .git
$ git init
$ git commit -am "Added C++ Boiler Plate"
```

The result is a fresh Git repository with one commit adding all files from the boiler plate. 

