project(project1)
cmake_minimum_required(VERSION 3.0.2)

# Project depends on Armadillo.
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})


# Set options common for all build types.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -Werror ")

# Somehow optimizations break an armadillo line in the code sporadically.
# Disabling all, and enabling everything in O1 seems to work.
# set(CMAKE_CXX_FLAGS_RELEASE "-O0 -fauto-inc-dec -fcompare-elim -fcprop-registers -fdce -fdefer-pop -fdse -fguess-branch-probability -fif-conversion2 -fif-conversion -fipa-pure-const -fipa-profile -fipa-reference -fmerge-constants -fsplit-wide-types -ftree-bit-ccp -ftree-builtin-call-dce -ftree-ccp -ftree-ch -ftree-copyrename -ftree-dce -ftree-dominator-opts -ftree-dse -ftree-forwprop -ftree-fre -ftree-phiprop -ftree-sra -ftree-pta -ftree-ter -funit-at-a-time")

# Compile VMC sources.
set(SOURCE_FILES "vmcsolver.cpp" "vmcimportancesolver.cpp")
add_library(engine ${SOURCE_FILES})

# Add executable run_mc.x
add_executable(run_mc.x run_mc.cpp)
add_dependencies(run_mc.x engine)
target_link_libraries(run_mc.x engine ${ARMADILLO_LIBRARIES})
