#include <gtest/gtest.h>

#include "definitions.hpp"
#include "system.hpp"
#include "simplegaussian.hpp"
#include "harmonicoscillatorhamiltonian.hpp"
#include "sampler.hpp"
#include "metropolissampler.hpp"

TEST(MetropolisSampler, basics) {
    const System init_system (10, 3);
    const SimpleGaussian psi(0.5, 1);
    const HarmonicOscillatorHamiltonian H_0;
    MetropolisSampler sampler (init_system, psi);

    System before = sampler.get_current_system();
    System after  = sampler.next_configuration();

    EXPECT_EQ(init_system.get_dimensions(), before.get_dimensions());
    EXPECT_EQ(init_system.get_n_bosons(), before.get_n_bosons());
    EXPECT_EQ(init_system.get_dimensions(), after.get_dimensions());
    EXPECT_EQ(init_system.get_n_bosons(), after.get_n_bosons());

    // One one particle should be attempted to move each time.
    // All other particles should remain unchanged, and the moving
    // particle shoudl be changed iff acceptance rate has increases.
    for (int runs = 1; runs < 10000; ++runs) {
        int moving = runs % init_system.get_n_bosons();

        Real ar = sampler.get_acceptance_rate();
        System before = sampler.get_current_system();
        System after  = sampler.next_configuration();

        for (int i = 0; i < init_system.get_n_bosons(); ++i) {
            if (i == moving) continue;
            ASSERT_EQ(before[i], after[i]);
        }

        if (sampler.get_acceptance_rate() > ar) {
            ASSERT_NE( before[moving], after[moving] );
        } else {
            ASSERT_EQ( before[moving], after[moving] );
        }
    }
}
