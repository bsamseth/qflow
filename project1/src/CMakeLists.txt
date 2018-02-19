project(project1)
cmake_minimum_required(VERSION 3.0.2)

# Project depends on Armadillo.
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})


# Set options common for all build types.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall")

# Compile VMC sources.
set(SOURCE_FILES "vmcsolver.cpp")
add_library(engine ${SOURCE_FILES})


# Add executable b_run.x
add_executable(b_run.x b_run.cpp)
add_dependencies(b_run.x engine)
target_link_libraries(b_run.x engine ${ARMADILLO_LIBRARIES})

