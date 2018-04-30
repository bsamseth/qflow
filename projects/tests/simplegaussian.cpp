#include <gtest/gtest.h>

#include "definitions.hpp"
#include "system.hpp"
#include "simplegaussian.hpp"


TEST(SimpleGaussian, zero_system) {
    System s (10, 3);
    SimpleGaussian psi ({0.5, 1, 0});

    // System all at origin.
    EXPECT_DOUBLE_EQ(1 , psi(s));
}

TEST(SimpleGaussian, call_with_beta) {
    System s (4, 3);
    s(0) = {{0.50833829,  0.07732213,  0.69294646}};
    s(1) = {{0.63837196,  0.48833327,  0.17570063}};
    s(2) = {{0.72436579,  0.36970369,  0.49771584}};
    s(3) = {{0.42984966,  0.72519657,  0.30454728}};

    SimpleGaussian psi ({0.5, 2.8});

    // Calculated by hand/calculator:
    Vector F_0 = {{-1.01667658, -0.15464426, -3.880500176}};
    EXPECT_DOUBLE_EQ(0.096971040514681181, psi(s));
    EXPECT_DOUBLE_EQ(-4.666685792898267, psi.derivative_alpha(s));
    for (int d = 0; d < s.get_dimensions(); ++d) {
        EXPECT_EQ(F_0[d], psi.drift_force(s, 0, d));
    }
}

TEST(SimpleGaussian, call_2D) {
    System s (3, 2);
    s(0) = {{1, 2}};
    s(1) = {{3, 4}};
    s(2) = {{5, 6}};

    // Beta shoudl not have any effect in 2D. Test both.
    SimpleGaussian psi_with_beta ({0.5, 2.8});
    SimpleGaussian psi ({0.5});

    // Calculated by hand/calculator:
    EXPECT_DOUBLE_EQ(1.7362052831002947e-20, psi_with_beta(s));
    EXPECT_DOUBLE_EQ(1.7362052831002947e-20, psi(s));
    EXPECT_DOUBLE_EQ(-91, psi_with_beta.derivative_alpha(s));
    EXPECT_DOUBLE_EQ(-91, psi.derivative_alpha(s));

    Vector F_2 = { {-10, -12} };
    for (int d = 0; d < s.get_dimensions(); ++d) {
        EXPECT_EQ(F_2[d], psi.drift_force(s, 2, d));
    }
}

