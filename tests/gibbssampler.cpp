#include "gibbssampler.hpp"

#include "definitions.hpp"
#include "optimizer.hpp"
#include "harmonicoscillator.hpp"
#include "rbmwavefunction.hpp"
#include "system.hpp"
#include "vector.hpp"

#include <gtest/gtest.h>

TEST(GibbsSampler, integrationTest)
{
    System                           init_system = System::Zero(2, 2);
    RBMWavefunction                  rbm(4, 2, 0.5, RBMWavefunction::GIBBS_FACTOR);
    HarmonicOscillator H;
    GibbsSampler                     sampler(init_system, rbm);
    SgdOptimizer                     sgd(0.9);

    H.optimize_wavefunction(rbm, sampler, 10000, 100, sgd, 0, false);

    Real E_L = H.local_energy(sampler, rbm, 1000);
    ASSERT_NEAR(2.0, E_L, 1e-3);
}
