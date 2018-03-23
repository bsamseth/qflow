#include <gtest/gtest.h>

#include "definitions.hpp"
#include "system.hpp"
#include "simplegaussian.hpp"
#include "harmonicoscillatorhamiltonian.hpp"
#include "sampler.hpp"
#include "importancesampler.hpp"

TEST(ImportanceSampler, reproducesExactInSimpleCase) {
    const Real E_TOL = 1e-3;
    const Real VAR_TOL = 1e-3;
    const int runs = 10000;
    const System init_system (2, 3);
    const SimpleGaussian psi(0.5, 1);
    const HarmonicOscillatorHamiltonian H_0;
    ImportanceSampler sampler (init_system, psi);

    System before = sampler.get_current_system();
    System after  = sampler.next_configuration();

    EXPECT_EQ(init_system.get_dimensions(), before.get_dimensions());
    EXPECT_EQ(init_system.get_n_bosons(), before.get_n_bosons());
    EXPECT_EQ(init_system.get_dimensions(), after.get_dimensions());
    EXPECT_EQ(init_system.get_n_bosons(), after.get_n_bosons());

    Real E  = 0;
    Real E2 = 0;
    // One one particle should be attempted to move each time.
    // All other particles should remain unchanged, and the moving
    // particle shoudl be changed iff acceptance rate has not decreased.
    for (int run = 1; run < runs; ++run) {
        int moving = run % init_system.get_n_bosons();

        Real ar = sampler.get_acceptance_rate();
        System before = sampler.get_current_system();
        System after  = sampler.next_configuration();

        for (int i = 0; i < init_system.get_n_bosons(); ++i) {
            if (i == moving) continue;
            ASSERT_EQ(before[i], after[i]);
        }

        if (sampler.get_acceptance_rate() >= ar) {
            ASSERT_NE( before[moving], after[moving] );
        } else {
            ASSERT_EQ( before[moving], after[moving] );
        }

        Real E_L = H_0.local_energy(after, psi);
        E += E_L;
        E2 += square(E_L);
    }

    E /= runs;
    E2 /= runs;

    EXPECT_NEAR(0.5 * init_system.get_n_bosons() * init_system.get_dimensions(), E, E_TOL);
    EXPECT_NEAR(0.0, E2 - square(E), VAR_TOL);
}
