#include <gtest/gtest.h>
#include <chrono>

#include "rbmharmonicoscillatorhamiltonian.hpp"
#include "metropolissampler.hpp"
#include "rbmwavefunction.hpp"

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
}





