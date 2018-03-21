#include <gtest/gtest.h>

#include "boson.hpp"

TEST(Boson, init) {
    Boson b1(1);
    Boson b3(3);
    b1[0] = 123.45;
    b3[0] = -1;
    b3[2] = 3;

    EXPECT_EQ(123.45, b1[0]);
    EXPECT_EQ(-1,     b3[0]);
    EXPECT_EQ( 0,     b3[1]);
    EXPECT_EQ( 3,     b3[2]);
}
