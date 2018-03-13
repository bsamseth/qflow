project(project1)
cmake_minimum_required(VERSION 3.0.2)

# Project depends on Armadillo.
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})


# Set options common for all build types.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -W -fopenmp")

# Compile VMC sources.
set(SOURCE_FILES "vmcsolver.cpp" "vmcimportancesolver.cpp")
add_library(engine ${SOURCE_FILES})

# Add executable run_mc.x
add_executable(run_mc.x run_mc.cpp)
add_dependencies(run_mc.x engine)
target_link_libraries(run_mc.x engine ${ARMADILLO_LIBRARIES})
