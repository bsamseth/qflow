// Bring in gtest
#include <gtest/gtest.h>
#include <cstdlib>
#include "mpiutil.hpp"

// Run all the tests that were declared with TEST()
int main(int argc, char **argv){
    std::srand(2018);
    mpiutil::initialize_mpi();
    testing::InitGoogleTest(&argc, argv);
    ::testing::TestEventListeners& listeners =
        ::testing::UnitTest::GetInstance()->listeners();
    if (mpiutil::get_rank() != 0) {
        delete listeners.Release(listeners.default_result_printer());
    }
    return RUN_ALL_TESTS();
}
