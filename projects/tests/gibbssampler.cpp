#include <gtest/gtest.h>

#include "definitions.hpp"
#include "vector.hpp"
#include "system.hpp"
#include "rbmwavefunction.hpp"
#include "rbmharmonicoscillatorhamiltonian.hpp"
#include "gibbssampler.hpp"

TEST(GibbsSampler, integrationTest) {
    System init_system(2, 2);
    RBMWavefunction rbm(4, 2, 0.5, RBMWavefunction::GIBBS_FACTOR);
    RBMHarmonicOscillatorHamiltonian H;
    GibbsSampler sampler(init_system, rbm);

    rbm.train(H, sampler, 10000, 100, 0.9, 0, false);

    Real E_L = 0;
    for (int i = 0; i < 1000; ++i) {
        E_L += H.local_energy(sampler.next_configuration(), rbm);
    }
    E_L /= 1000;

    ASSERT_NEAR(2.0, E_L, 1e-3);
}
