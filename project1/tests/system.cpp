#include <gtest/gtest.h>

#include "system.hpp"

TEST(System, basics) {
    System s (10, 3);

    s[0][0] = -123;
    s[0][2] = 123;

    EXPECT_EQ(-123, s[0][0]);
    EXPECT_EQ( 123, s[0][2]);
    EXPECT_EQ( 10, s.get_n_bosons() );
    EXPECT_EQ( 3 , s.get_dimensions() );

    System ss = s;

    EXPECT_EQ(-123, ss[0][0]);
    EXPECT_EQ( 123, ss[0][2]);
    EXPECT_EQ( 10, ss.get_n_bosons() );
    EXPECT_EQ( 3 , ss.get_dimensions() );
}
