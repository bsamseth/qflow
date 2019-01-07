#include <gtest/gtest.h>
#include <chrono>

#include "system.hpp"

TEST(System, basics) {
    System s (3, 10);

    s(0, 0) = -123;
    s(0, 2) = 123;

    EXPECT_EQ(-123, s(0, 0));
    EXPECT_EQ( 123, s(0, 2));
    EXPECT_EQ( 10, s.cols() );
    EXPECT_EQ( 3 , s.rows() );

    System ss = s;

    EXPECT_EQ(-123, ss(0, 0));
    EXPECT_EQ( 123, ss(0, 2));
    EXPECT_EQ( 10, ss.cols() );
    EXPECT_EQ( 3 , ss.rows() );
}

TEST(System, distances) {
    System s (3, 4);
    s.col(0) << 1, 2, 3;
    s.col(1) << -2, 0, 1;
    s.col(2) << -4, -3, -1;
    s.col(3) << 10, 2, 0;

    std::vector<std::vector<Real>> expected = {
        {0.0000000000000000, 4.1231056256176606, 8.1240384046359608, 9.4868329805051381},
        {4.1231056256176606, 0.0000000000000000, 4.1231056256176606, 12.2065556157337021},
        {8.1240384046359608, 4.1231056256176606, 0.0000000000000000, 14.8996644257513395},
        {9.4868329805051381, 12.2065556157337021, 14.8996644257513395, 0.0000000000000000},
    };


    for (int i = 0; i < (int) expected.size(); i++) {
        for (int j = 0; j < (int) expected.size(); j++) {
            ASSERT_DOUBLE_EQ(expected[i][j], distance(s, i, j));;
        }
    }
}
