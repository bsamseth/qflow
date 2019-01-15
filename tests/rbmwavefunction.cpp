#include <gtest/gtest.h>
#include <string>

#include "prettyprint.hpp"
#include "metropolissampler.hpp"
#include "importancesampler.hpp"
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
    public:
        constexpr static int P = 2;
        constexpr static int D = 3;
        constexpr static int M = P * D;
        constexpr static int N = M + 1;

        System *s;
        RBMWavefunction *rbm;

        virtual void SetUp() {
            s = new System(3, 2);
            s->col(0) << -0.5009136425144746,  0.0118290278182348,  1.0861838035029527;
            s->col(1) << -0.1180658328483096, -1.0091187557423571, -0.5365339555565911;

            rbm = new RBMWavefunction(M, N, 2);
            rbm->set_parameters({-0.7665546243072233,  0.9406054576757206, -0.1411666581066846,
                                   2.3698247761289206, -0.0925344556191133, -0.822443320890463 ,
                                  -0.3367728130900145,  0.6042382684274119,  1.1031676900053826,
                                  -1.11343573670209  , -1.2698025674851516,  0.6405021942562454,
                                   1.1083481552380658,  0.9132139881362205, -0.5289324083374893,
                                  -0.6259702462677719,  0.4288389687300546, -0.2275709099552632,
                                   0.0800292163048842, -0.2006399070516838,  2.1892037314628694,
                                  -0.1135711243410274, -0.7075746186364059, -0.9951770429084715,
                                   0.0992837432654784, -1.1374021103825422, -0.2257842764845613,
                                   1.0681137462003307,  0.4475426303688199,  1.2644501323447888,
                                   0.8874277315102514,  0.9465699704550599, -2.0148512749309386,
                                  -0.4172389063717908,  0.8106036574891917,  1.0111662774773056,
                                  -0.2573547886564483,  0.3167544620067116,  0.0489496271535557,
                                  -0.177898745022331 ,  0.4915684273366239, -0.7627213279660949,
                                   0.4743876191911438,  0.8256348759803365,  0.3608972886871439,
                                   1.0230129736402584,  0.4995386943252447, -0.9389995079735179,
                                  -0.1790977108647215, -0.0797636435834231, -0.6916788192310428,
                                  -0.0363434391489085, -1.2944879136187115,  0.8263967557636448,
                                  -0.0240516420501065});
        }
        virtual void TearDown() {
            delete s;
            delete rbm;
        }
};

TEST_F(RBMWavefunctionTest, init) {
    ASSERT_EQ(M + N + M*N, rbm->get_parameters().size());
    ASSERT_DOUBLE_EQ(-0.7665546243072233, rbm->get_parameters()[rbm->a(0)]);
    ASSERT_DOUBLE_EQ(-0.822443320890463, rbm->get_parameters()[rbm->a(M-1)]);
    ASSERT_DOUBLE_EQ(-0.3367728130900145, rbm->get_parameters()[rbm->b(0)]);
    ASSERT_DOUBLE_EQ(1.1083481552380658, rbm->get_parameters()[rbm->b(N-1)]);
    ASSERT_DOUBLE_EQ(0.9132139881362205, rbm->get_parameters()[rbm->w(0, 0)]);
    ASSERT_DOUBLE_EQ(-0.0240516420501065, rbm->get_parameters()[rbm->w(M-1, N-1)]);
}

TEST_F(RBMWavefunctionTest, evaluation) {
    // Values calculated by hand/calculator.
    EXPECT_DOUBLE_EQ(62.977993642663755, rbm->operator()(*s));
}

TEST_F(RBMWavefunctionTest, gradient) {
    RowVector gradient = rbm->gradient(*s);
    RowVector expected_a = vector_from_sequence({0.1328204908963744, -0.4643882149287429,  0.6136752308048187,
                          -1.2439453044886151, -0.4582921500616219,  0.142954682666936});
    RowVector expected_b = vector_from_sequence({0.6017040564983215,  0.6685085554779254,  0.8489820123505009,
                          0.2817462931894426,  0.2952675674909326,  0.2801418147825886,
                          0.7992927449865912});
    RowVector expected_w = vector_from_sequence({-0.1507008853281547045 , -0.16743252778826867022,
                          -0.21263333611787901822, -0.07056528099323740311,
                          -0.07395177637413576088, -0.07016342843168088539,
                          -0.20018832016331319923,  0.00355878701133167767,
                           0.00395390314973816364,  0.00502131592063750822,
                           0.00166639236991122409,  0.00174636413483637842,
                           0.00165690266005700778,  0.00472742805767981198,
                           0.32678060033525119232,  0.36306158273163885442,
                           0.46107525564022888709,  0.153014130179683433  ,
                           0.16035742475418299224,  0.15214275095038592989,
                           0.43408941693092562009, -0.03552034527934033042,
                          -0.03946400968436082624, -0.0501178841806978817 ,
                          -0.01663230537866778116, -0.01743050563445571605,
                          -0.0165375883389716101 , -0.04718458181322671835,
                          -0.30359542440935755048, -0.33730226085350228926,
                          -0.4283618359753899596 , -0.14215773440917583992,
                          -0.14898002015876118942, -0.14134817978240585279,
                          -0.40329065014738102457, -0.16141732875374550926,
                          -0.17933876979699708842, -0.22775383864140447132,
                          -0.07558322657416934043, -0.07921053796674140668,
                          -0.07515279800105209573, -0.214423849057670729});
    for (int i = 0; i < M; ++i)
        EXPECT_DOUBLE_EQ(expected_a[i], gradient[rbm->a(i)]);
    for (int j = 0; j < N; ++j)
        EXPECT_DOUBLE_EQ(expected_b[j], gradient[rbm->b(j)]);
    // d/dw gives some very minor differences from expected values.
    // This is so small that it is with all likelihood rounding errors
    // in both calculated values.
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            EXPECT_NEAR(expected_w[i * N + j], gradient[rbm->w(i, j)], 1e-16);
}

TEST_F(RBMWavefunctionTest, laplacian) {
    EXPECT_DOUBLE_EQ(3.2384829101357431, rbm->laplacian(*s));
}

TEST_F(RBMWavefunctionTest, RBMHarmonicOscillatorHamiltonian) {
    RBMHarmonicOscillatorHamiltonian H;
    EXPECT_DOUBLE_EQ(1.3754892738453772, H.external_potential(*s));
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
        System s (D, P);
        RBMWavefunction rbm(M, N, sigma2);
        rbm.set_parameters(RowVector::Zero(M + N + M * N));

        // Random system config.
        for (int i = 0; i < D; ++i) {
            for (int j = 0; j < P; j++)
                s(i, j) = double_gen();
        }

        Real expected = 0.5 * P * D;
        ASSERT_NEAR(expected, H.local_energy(s, rbm), expected * 1e-15);
        ASSERT_NEAR(expected, H.local_energy_numeric(s, rbm), expected * 1e-6);
    }
}


TEST(RBMWavefunction, trainSimpleCase3D) {
    System init_system (3, 2);
    RBMWavefunction rbm (6, 2);
    ImportanceSampler sampler (init_system, rbm, 0.5);
    RBMHarmonicOscillatorHamiltonian H;
    AdamOptimizer sgd(rbm.get_parameters().size());

    rbm.train(H, sampler, 5000, 100, sgd, 0.0, false);

    Real E_L = 0;
    for (int i = 0; i < 1000; ++i)
        E_L += H.local_energy(sampler.next_configuration(), rbm);
    E_L /= 1000;

    ASSERT_NEAR(3.0, E_L, 1e-3);
}

