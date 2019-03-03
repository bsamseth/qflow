#include "system.hpp"

#include <chrono>
#include <gtest/gtest.h>

TEST(System, basics)
{
    System s(10, 3);

    s(0, 0) = -123;
    s(0, 2) = 123;

    EXPECT_EQ(-123, s(0, 0));
    EXPECT_EQ(123, s(0, 2));
    EXPECT_EQ(10, s.rows());
    EXPECT_EQ(3, s.cols());

    System ss = s;

    EXPECT_EQ(-123, ss(0, 0));
    EXPECT_EQ(123, ss(0, 2));
    EXPECT_EQ(10, ss.rows());
    EXPECT_EQ(3, ss.cols());

    EXPECT_EQ(123, s.data()[2]);  // Check rowmajor storage order.
}

TEST(System, distances)
{
    System s(4, 3);
    s.row(0) << 1, 2, 3;
    s.row(1) << -2, 0, 1;
    s.row(2) << -4, -3, -1;
    s.row(3) << 10, 2, 0;

    std::vector<std::vector<Real>> expected = {
        {0.0000000000000000,
         4.1231056256176606,
         8.1240384046359608,
         9.4868329805051381},
        {4.1231056256176606,
         0.0000000000000000,
         4.1231056256176606,
         12.2065556157337021},
        {8.1240384046359608,
         4.1231056256176606,
         0.0000000000000000,
         14.8996644257513395},
        {9.4868329805051381,
         12.2065556157337021,
         14.8996644257513395,
         0.0000000000000000},
    };

    for (int i = 0; i < (int) expected.size(); i++)
    {
        for (int j = 0; j < (int) expected.size(); j++)
        {
            ASSERT_DOUBLE_EQ(expected[i][j], distance(s, i, j));
            ;
        }
    }
}
