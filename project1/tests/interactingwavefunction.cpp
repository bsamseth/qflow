#include <gtest/gtest.h>

#include "definitions.hpp"
#include "system.hpp"
#include "simplegaussian.hpp"
#include "interactingwavefunction.hpp"

TEST(InteractingWavefunction, zero_system) {
    System s (10, 3);
    InteractingWavefunction psi (0.5, 1, 0);

    // System all at origin - impossible for interacting case.
    // Psi, and d(Psi)/d(alpha) should be zero, due to contribution from correlation.
    // This is true regardless of the value of a.
    EXPECT_DOUBLE_EQ(0, psi(s));
    EXPECT_DOUBLE_EQ(0, psi.derivative_alpha(s));
    EXPECT_DOUBLE_EQ(0, psi.correlation(s));
}

TEST(InteractingWavefunction, call_with_beta) {
    System s (4, 3);
    s[0] = {{0.50833829,  0.07732213,  0.69294646}};
    s[1] = {{0.63837196,  0.48833327,  0.17570063}};
    s[2] = {{0.72436579,  0.36970369,  0.49771584}};
    s[3] = {{0.42984966,  0.72519657,  0.30454728}};

    InteractingWavefunction psi_1 (0.5, 2.8, 0.0043);
    InteractingWavefunction psi_2 (0.5, 2.8, 0.);
    SimpleGaussian psi (0.5, 2.8);

    // With a set to zero, this system should behave as a simple gaussian wavefunc.
    EXPECT_DOUBLE_EQ(psi(s), psi_2(s));
    EXPECT_DOUBLE_EQ(psi.derivative_alpha(s), psi_2.derivative_alpha(s));

    // Calculated by hand/calculator:
    EXPECT_DOUBLE_EQ(0.94543129358834554 * psi(s), psi_1(s));
    EXPECT_DOUBLE_EQ(0.94543129358834554 * psi.derivative_alpha(s), psi_1.derivative_alpha(s));
}
