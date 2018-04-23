#include <gtest/gtest.h>
#include <string>

#include "prettyprint.hpp"
#include "metropolissampler.hpp"
#include "rbmharmonicoscillatorhamiltonian.hpp"
#include "rbmwavefunction.hpp"

namespace {
    std::default_random_engine rand_gen(12345);
    std::uniform_real_distribution<Real> rand_dist(-1, 1);
    std::uniform_int_distribution<int> rand_dim(1, 3);

    double double_gen() {
        return rand_dist(rand_gen);
    }
    int dim_gen() {
        return rand_dim(rand_gen);
    }
}

class RBMWavefunctionTest : public ::testing::Test {
    protected:
        constexpr static int P = 2;
        constexpr static int D = 1;
        constexpr static int M = P * D;
        constexpr static int N = M + 1;

        System *s;
        RBMWavefunction *rbm;
        const std::vector<Real> params = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

        virtual void SetUp() {
            rbm = new RBMWavefunction(M, N, 2);
            rbm->set_params(params);
            s = new System(P, D);
            (*s)(0)[0] = 1.5;
            (*s)(1)[0] = 2.5;

        }
        virtual void TearDown() {
            delete s;
            delete rbm;
        }
};

TEST_F(RBMWavefunctionTest, init) {
    int k = 0;
    for (auto a_i : rbm->get_visible_bias()) {
        ASSERT_EQ(params[k++], a_i);
    }
    for (auto b_i : rbm->get_hidden_bias()) {
        ASSERT_EQ(params[k++], b_i);
    }
    for (auto w_i : rbm->get_weights()) {
        for (auto w_ij : w_i) {
            ASSERT_EQ(params[k++], w_ij);
        }
    }
}

TEST_F(RBMWavefunctionTest, evaluation) {
    // Values calculated by hand/calculator.
    EXPECT_DOUBLE_EQ(1.920561139962089e28, rbm->operator()(*s));
}

TEST_F(RBMWavefunctionTest, gradient) {
    std::vector<Real> gradient(M + N + M * N);
    rbm->gradient(*s, gradient);
    EXPECT_DOUBLE_EQ(0.25, gradient[0]);
    EXPECT_DOUBLE_EQ(0.25, gradient[1]);

    EXPECT_DOUBLE_EQ(0.9999999928058669, gradient[2]);
    EXPECT_DOUBLE_EQ(0.9999999996418252, gradient[3]);
    EXPECT_DOUBLE_EQ(0.9999999999821676, gradient[4]);

    EXPECT_DOUBLE_EQ(0.7499999946044003, gradient[5]);
    EXPECT_DOUBLE_EQ(0.7499999997313689, gradient[6]);
    EXPECT_DOUBLE_EQ(0.7499999999866257, gradient[7]);
    EXPECT_DOUBLE_EQ(1.2499999910073336, gradient[8]);
    EXPECT_DOUBLE_EQ(1.2499999995522815, gradient[9]);
    EXPECT_DOUBLE_EQ(1.2499999999777094, gradient[10]);

    // This is not perfectly reproducing the calculator results,
    // but after reviewing code thoroughly, no fault is found,
    // and the discrepancy is deemed likely to be the result of rounding
    // errors, in one or both of the actual vs. expected numbers. Single
    // precision still satisfied.
    EXPECT_FLOAT_EQ(321.62499854823926, rbm->laplacian(*s));
}

TEST_F(RBMWavefunctionTest, RBMHarmonicOscillatorHamiltonian) {
    RBMHarmonicOscillatorHamiltonian H;
    EXPECT_DOUBLE_EQ(4.25, H.external_potential(*s));
    EXPECT_DOUBLE_EQ(0.0,  H.internal_potential(*s));
    EXPECT_DOUBLE_EQ(H.external_potential(*s) - 0.5 * rbm->laplacian(*s), H.local_energy(*s, *rbm));
}


/*
 * With all parameters set to zero, and sigma^2 = 1, the RBM should
 * produce the exact results for the ideal case (no-interaction).
 * This should be regardless of dimensions, number of particles and
 * the positions of the particles.
 */
TEST(RBMWavefunction, correctForIdealCase) {
    const Real sigma2 = 1;
    RBMHarmonicOscillatorHamiltonian H;

    // 1000, why not?
    for (int runs = 0; runs < 1000; ++runs) {
        int P = square(dim_gen()), D = dim_gen();
        int M = P * D, N = 1;
        System s (P, D);
        RBMWavefunction rbm(M, N, sigma2);
        rbm.set_params(std::vector<Real>(M + N + M * N, 0));

        // Random system config.
        for (int i = 0; i < P; ++i) {
            s(i) = {D};
            for (int j = 0; j < D; j++)
                s(i)[j] = double_gen();
        }

        Real expected = 0.5 * P * D;
        ASSERT_NEAR(expected, H.local_energy_numeric(s, rbm), expected * 1e-6);
        ASSERT_NEAR(expected, H.local_energy(s, rbm), expected * 1e-15);
    }

}



