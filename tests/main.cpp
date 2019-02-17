// Bring in gtest
#include <gtest/gtest.h>
#include <cstdlib>
#include "mpiutil.hpp"

// Run all the tests that were declared with TEST()
int main(int argc, char **argv){
    std::srand(2018);
    mpiutil::initialize_mpi();
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
