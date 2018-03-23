#include <gtest/gtest.h>
#include <random>
#include <limits>

#include "definitions.hpp"
#include "system.hpp"
#include "interactingwavefunction.hpp"
#include "interactinghamiltonian.hpp"

namespace {
    std::default_random_engine rand_gen(12345);
    std::uniform_real_distribution<Real> rand_dist(-1, 1);
    std::uniform_int_distribution<int> rand_dim(1, 3);

    auto double_gen = std::bind(rand_dist, rand_gen);
    auto dim_gen    = std::bind(rand_dim, rand_gen);
}

class InteractingHamiltonianTest : public ::testing::Test {
    protected:
        const Real a = 0.0043;
        System *s1;
        System *s2;
        InteractingHamiltonian H_1 {1, a};
        InteractingHamiltonian H_2 {2.8, a};
        InteractingHamiltonian H_3 {2.8, 0.0};

        virtual void SetUp() {
            s1 = new System(4, 3);
            s2 = new System(5, 3);
            // Random values.
            (*s1)[0] = {{0.50833829,  0.07732213,  0.69294646}};
            (*s1)[1] = {{0.63837196,  0.48833327,  0.17570063}};
            (*s1)[2] = {{0.72436579,  0.36970369,  0.49771584}};
            (*s1)[3] = {{0.42984966,  0.72519657,  0.30454728}};


            // Random values.
            (*s2)[0] = {{0.50833829,  0.07732213,  0.69294646}};
            (*s2)[1] = {{0.63837196,  0.48833327,  0.17570063}};
            (*s2)[2] = {{0.72436579,  0.36970369,  0.49771584}};
            (*s2)[3] = {{0.42984966,  0.72519657,  0.30454728}};
            (*s2)[4] = {{0.50833829,  0.07732212,  0.69294646}};  // almost s2[0]
        }
        virtual void TearDown() {
            delete s1;
            delete s2;
        }
};

TEST_F(InteractingHamiltonianTest, potential) {
    // Result should be agnostic to omega_z value
    EXPECT_DOUBLE_EQ(0, H_1.internal_potential(*s1));
    EXPECT_DOUBLE_EQ(0, H_2.internal_potential(*s1));
    EXPECT_DOUBLE_EQ(0, H_3.internal_potential(*s1));
    EXPECT_DOUBLE_EQ(std::numeric_limits<Real>::max(), H_1.internal_potential(*s2));
    EXPECT_DOUBLE_EQ(std::numeric_limits<Real>::max(), H_2.internal_potential(*s2));
    EXPECT_DOUBLE_EQ(0, H_3.internal_potential(*s2));
}

TEST_F(InteractingHamiltonianTest, localEnergy) {
    const Real gamma = 2.8;
    SimpleGaussian psi_0(0.5, gamma, 0);
    InteractingWavefunction psi_T(0.5, gamma, a);
    HarmonicOscillatorHamiltonian H_0(gamma);

    // With a = 0, no difference to the simple case shoudl be observed.
    EXPECT_DOUBLE_EQ(H_0.local_energy(*s1, psi_0), H_3.local_energy(*s1, psi_0));
    EXPECT_DOUBLE_EQ(H_0.local_energy(*s2, psi_0), H_3.local_energy(*s2, psi_0));

    // With a > 0, compare with numeric. For s2, V_int = inf, so both should be very equal.
    // For s1, expect decent numerical accuracy.
    EXPECT_NEAR(H_1.local_energy_numeric(*s1, psi_T), H_1.local_energy(*s1, psi_T), 5e-6);
    EXPECT_DOUBLE_EQ(H_1.local_energy_numeric(*s2, psi_T), H_1.local_energy(*s2, psi_T));
    EXPECT_NEAR(H_2.local_energy_numeric(*s1, psi_T), H_2.local_energy(*s1, psi_T), 5e-6);
    EXPECT_DOUBLE_EQ(H_2.local_energy_numeric(*s2, psi_T), H_2.local_energy(*s2, psi_T));
}

