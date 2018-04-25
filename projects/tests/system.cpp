#include <gtest/gtest.h>
#include <chrono>

#include "system.hpp"

TEST(System, basics) {
    System s (10, 3);

    s(0)[0] = -123;
    s(0)[2] = 123;

    EXPECT_EQ(-123, s[0][0]);
    EXPECT_EQ( 123, s[0][2]);
    EXPECT_EQ( 10, s.get_n_particles() );
    EXPECT_EQ( 3 , s.get_dimensions() );

    System ss = s;

    EXPECT_EQ(-123, ss[0][0]);
    EXPECT_EQ( 123, ss[0][2]);
    EXPECT_EQ( 10, ss.get_n_particles() );
    EXPECT_EQ( 3 , ss.get_dimensions() );
}

TEST(System, distances) {
    System s (4, 3);
    s(0) = {{1, 2, 3}};
    s(1) = {{-2, 0, 1}};
    s(2) = {{-4, -3, -1}};
    s(3) = {{10, 2, 0}};

    std::vector<std::vector<Real>> expected = {
        {0.0000000000000000, 4.1231056256176606, 8.1240384046359608, 9.4868329805051381},
        {4.1231056256176606, 0.0000000000000000, 4.1231056256176606, 12.2065556157337021},
        {8.1240384046359608, 4.1231056256176606, 0.0000000000000000, 14.8996644257513395},
        {9.4868329805051381, 12.2065556157337021, 14.8996644257513395, 0.0000000000000000},
    };


    for (int k = 0; k < 100; ++k) {
        for (int i = 0; i < (int) expected.size(); i++) {
            for (int j = 0; j < (int) expected.size(); j++) {
                ASSERT_DOUBLE_EQ(expected[i][j], s.distance(i, j));
                ASSERT_FALSE(s.get_dirty()[i][j]);
            }
        }

        // Simply accessing the system should use the const version,
        // and distances should therefore not have to recalculated.
        // Try to touch all bosons, and veryfy that
        Real sum = 0;
        for (int i = 0; i < (int) expected.size(); i++) {
            for (int j = 0; j < (int) expected.size(); j++) {
                sum += 2 * s[i][j] - 3 * s[j][i];
            }
        }

        for (int i = 0; i < (int) expected.size(); ++i) {
            for (int j = 0; j < (int) expected.size(); ++j) {
                ASSERT_FALSE(s.get_dirty()[i][j]);
            }
        }

        // Changing a boson should trigger a recalculation, even if changed
        // to the existing value;
        int a = k % s.get_n_particles();
        s(a) = s[a];

        for (int i = 0; i < (int) expected.size(); ++i) {
            for (int j = 0; j < (int) expected.size(); ++j) {
                ASSERT_EQ(i == a or j == a, s.get_dirty()[i][j]);
                ASSERT_EQ(i == a or j == a, s.get_dirty()[j][i]);
            }
        }
    }
}
