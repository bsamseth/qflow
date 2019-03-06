cmake_minimum_required(VERSION 3.1.3)

project(VMC VERSION 0.1.0.0 LANGUAGES C CXX)

# Include stuff.
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
include(ConfigSafeGuards)
include(Colors)



# --------------------------------------------------------------------------------
#                          Compile flags (change as needed).
# --------------------------------------------------------------------------------
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

# Things to always include as flags. Change as needed.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -fopenmp")

# Build-type specific flags. Change as needed.
SET(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
SET(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Building with the following extra flags: ${CMAKE_CXX_FLAGS}")


# --------------------------------------------------------------------------------
#                         Locate files (no change needed).
# --------------------------------------------------------------------------------
find_package(MPI REQUIRED)

# We make sure that CMake sees all the files.
include_directories(
    ${PROJECT_SOURCE_DIR}/qflow
    ${PROJECT_SOURCE_DIR}/qflow/wavefunctions
    ${PROJECT_SOURCE_DIR}/qflow/wavefunctions/nn
    ${PROJECT_SOURCE_DIR}/qflow/wavefunctions/nn/layers
    ${PROJECT_SOURCE_DIR}/qflow/wavefunctions/nn/activations
    ${PROJECT_SOURCE_DIR}/qflow/hamiltonians
    ${PROJECT_SOURCE_DIR}/qflow/samplers
    ${PROJECT_SOURCE_DIR}/qflow/optimizers
    ${PROJECT_SOURCE_DIR}/external/googletest
    ${PROJECT_SOURCE_DIR}/external/googletest/include
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${PROJECT_SOURCE_DIR}/external/eigen/Eigen
    ${MPI_INCLUDE_PATH})


set(SOURCES
    qflow/definitions.cpp
    qflow/hamiltonians/coulombharmonicoscillator.cpp
    qflow/hamiltonians/hamiltonian.cpp
    qflow/hamiltonians/hardsphereharmonicoscillator.cpp
    qflow/hamiltonians/harmonicoscillator.cpp
    qflow/mpiutil.cpp
    qflow/optimizers/optimizer.cpp
    qflow/samplers/gibbssampler.cpp
    qflow/samplers/importancesampler.cpp
    qflow/samplers/metropolissampler.cpp
    qflow/samplers/sampler.cpp
    qflow/wavefunctions/hardspherewavefunction.cpp
    qflow/wavefunctions/jastroworion.cpp
    qflow/wavefunctions/nn/activations/activation.cpp
    qflow/wavefunctions/nn/dnn.cpp
    qflow/wavefunctions/nn/layers/layer.cpp
    qflow/wavefunctions/rbmsymmetricwavefunction.cpp
    qflow/wavefunctions/rbmwavefunction.cpp
    qflow/wavefunctions/simplegaussian.cpp
    qflow/wavefunctions/wavefunction.cpp
    qflow/wavefunctions/wavefunctionpooling.cpp
    qflow/wavefunctions/wavefunctionproduct.cpp
)
set(PYSOURCES
    qflow/hamiltonians/pyhamiltonian.cpp
    qflow/optimizers/pyoptimizer.cpp
    qflow/pymain.cpp
    qflow/samplers/pysampler.cpp
    qflow/wavefunctions/nn/activations/pyactivation.cpp
    qflow/wavefunctions/nn/layers/pylayer.cpp
    qflow/wavefunctions/nn/pynn.cpp
    qflow/wavefunctions/pywavefunction.cpp
)

file(GLOB TESTFILES "tests/*.cpp")
set(TEST_MAIN unit_tests.x)  # Default name for test executable (change if you wish).



# --------------------------------------------------------------------------------
#                            Build! (Change as needed)
# --------------------------------------------------------------------------------
# Compile all sources into a library. Called engine here (change if you wish).
add_library( engine SHARED ${SOURCES})
target_link_libraries(engine PRIVATE ${MPI_LIBRARIES})

# Compile Python bindings.
add_subdirectory(external)  # Includes external/pybind
pybind11_add_module(_qflow_backend SHARED ${PYSOURCES})
target_link_libraries(_qflow_backend PRIVATE engine)

if(MPI_COMPILE_FLAGS)
  set_target_properties(engine PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
  set_target_properties(_qflow_backend PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  set_target_properties(engine PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
  set_target_properties(_qflow_backend PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()
# --------------------------------------------------------------------------------
#                         Make Tests (no change needed).
# --------------------------------------------------------------------------------
# Add a make target 'gtest', that runs the tests (and builds all dependencies).
# The setup of Google Test is done at the very end of this file.
add_executable(${TEST_MAIN} ${TESTFILES})
add_dependencies(${TEST_MAIN} googletest engine)
target_link_libraries(${TEST_MAIN} googletest engine pthread ${MPI_LIBRARIES})
add_custom_target(gtest
    COMMAND "${PROJECT_BINARY_DIR}/${TEST_MAIN}"
    DEPENDS engine ${TEST_MAIN})


# Add a standard make target 'test' that runs the tests under CTest (only as an alt. to gtest).
include(CTest)
enable_testing()
add_test(unit_tests ${PROJECT_BINARY_DIR}/${TEST_MAIN})

# --------------------------------------------------------------------------------
#                         Coverage (no change needed).
# --------------------------------------------------------------------------------
include(CodeCoverage)

# --------------------------------------------------------------------------------
#                         Documentation (no change needed).
# --------------------------------------------------------------------------------
# Add a make target 'doc' to generate API documentation with Doxygen.
# You should set options to your liking in the file 'Doxyfile.in'.
find_package(Doxygen)
if(DOXYGEN_FOUND)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
    add_custom_target(doc
        ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile &> doxygen.log
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen" VERBATIM
        )
endif(DOXYGEN_FOUND)


# --------------------------------------------------------------------------------
#                         Google Test (no change needed).
# --------------------------------------------------------------------------------
# The following makes sure that an up-to-date version of googletest is available,
# and built so that it may be used by your tests.
add_custom_target( git_update
    COMMAND git submodule init
    COMMAND git submodule update
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} )
add_library( googletest
    ${PROJECT_SOURCE_DIR}/external/googletest/src/gtest-all.cc
    ${PROJECT_SOURCE_DIR}/external/googletest/src/gtest_main.cc )
add_dependencies(googletest git_update)
set_source_files_properties(${PROJECT_SOURCE_DIR}/external/googletest/src/gtest-all.cc  PROPERTIES GENERATED 1)
set_source_files_properties(${PROJECT_SOURCE_DIR}/external/googletest/src/gtest_main.cc PROPERTIES GENERATED 1)


# --------------------------------------------------------------------------------
#                            Misc (no change needed).
# --------------------------------------------------------------------------------
# Ask CMake to output a compile_commands.json file for YCM.
set( CMAKE_EXPORT_COMPILE_COMMANDS 1 )

IF( EXISTS "${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json" )
    message(STATUS "Exporting compile_commands.json to source directory.")
    EXECUTE_PROCESS( COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json
        ${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json
        )
ELSE()
    message(STATUS "compile_commands.json not exported. Rerun cmake.")
ENDIF()
