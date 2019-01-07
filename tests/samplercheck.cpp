#include <gtest/gtest.h>

#include "definitions.hpp"
#include "system.hpp"
#include "simplegaussian.hpp"
#include "harmonicoscillatorhamiltonian.hpp"
#include "sampler.hpp"
#include "metropolissampler.hpp"
#include "importancesampler.hpp"

#define METRO_SAMPLER 1
#define IMPO_SAMPLER 2

void sampler_sanity(int sampler_type) {
    const int runs = 10000;
    const System init_system (10, 3);
    const SimpleGaussian psi({0.5, 1});
    const HarmonicOscillatorHamiltonian H_0;

    Sampler *sp;
    if (sampler_type == METRO_SAMPLER)
        sp = new MetropolisSampler(init_system, psi);
    else
        sp = new ImportanceSampler(init_system, psi);
    Sampler &sampler = *sp;

    System before = sampler.get_current_system();
    System after  = sampler.next_configuration();

    EXPECT_EQ(init_system.rows(), before.rows());
    EXPECT_EQ(init_system.cols(), before.cols());
    EXPECT_EQ(init_system.rows(), after.rows());
    EXPECT_EQ(init_system.cols(), after.cols());

    // One one particle should be attempted to move each time.
    // All other particles should remain unchanged, and the moving
    // particle should be changed iff acceptance rate has not decreased.
    for (int run = 1; run <= runs; ++run) {
        int moving = run % init_system.cols();

        long accepted = sampler.get_accepted_steps();
        long total = sampler.get_total_steps();
        System before = sampler.get_current_system();
        System after  = sampler.next_configuration();

        ASSERT_TRUE(total == sampler.get_total_steps() - 1);

        for (int i = 0; i < init_system.cols(); ++i) {
            if (i == moving) continue;
            ASSERT_TRUE(before.col(i).isApprox(after.col(i)));
        }

        if (sampler.get_accepted_steps() > accepted) {
            ASSERT_FALSE( before.col(moving).isApprox(after.col(moving)) );
        } else {
            ASSERT_TRUE( before.col(moving).isApprox(after.col(moving)) );
        }

        Real E_L = H_0.local_energy(after, psi);

        // Energy should still be totaly stable for the exact case.
        // Minor, minor rounding error might be here, so every so often a
        // ASSERT_DOUBLE_EQ will fail due to being 5 ULP different, where 4 ULP (unit in the last place)
        // is Google tests limit. This is still good enough. ASSERT_FLOAT_EQ works.
        ASSERT_NEAR(init_system.rows() * init_system.cols() * 0.5, E_L, 1e-13);
    }
}

void sampler_integration_test(int sampler_type, Real expected_acceptance, Real analytic_tol, Real numeric_tol) {
    const Real gamma = 2.82843;
    const Real alpha = 0.5;
    const int runs = 10000;
    const System init_system (10, 3);
    const SimpleGaussian psi({alpha, gamma});
    const HarmonicOscillatorHamiltonian H(gamma);

    Sampler *sp;
    if (sampler_type == METRO_SAMPLER)
        sp = new MetropolisSampler(init_system, psi);
    else
        sp = new ImportanceSampler(init_system, psi);
    Sampler &sampler = *sp;

    Real E_analytic = 0;
    Real E_numeric = 0;

    // Thermalize (adjusts for any random bad starting positions.)
    for (int run = 0; run < runs; run++)
        sampler.next_configuration();

    for (int run = 0; run < runs; ++run) {
        System &state = sampler.next_configuration();

        E_analytic += H.local_energy(state, psi);
        E_numeric += H.local_energy_numeric(state, psi);
    }

    E_analytic /= runs;
    E_numeric /= runs;

    // Acceptance rate, reflecting only after thermalization.
    auto steps = sampler.get_total_steps();
    Real ar = sampler.get_acceptance_rate() * steps / (steps - runs);

    // Check that acceptance rate has not dropped below what has been seen before.
    // Just test it, in case we muck something up that decreases it later on.
    EXPECT_GE(ar, expected_acceptance);

    EXPECT_NEAR(H.gross_pitaevskii_energy(init_system, psi), E_analytic, analytic_tol) << "ar = " << ar;
    EXPECT_NEAR(H.gross_pitaevskii_energy(init_system, psi), E_numeric, numeric_tol) << "ar = " << ar;
}

TEST(MetropolisSampler, sanityCheck) {
    sampler_sanity(METRO_SAMPLER);
}
TEST(ImportanceSampler, sanityCheck) {
    sampler_sanity(IMPO_SAMPLER);
}
TEST(MetropolisSampler, integrationTest) {
    sampler_integration_test(METRO_SAMPLER, 0.64, 1e-12, 5e-6);
}
TEST(ImportanceSampler, integrationTest) {
    sampler_integration_test(IMPO_SAMPLER, 0.95, 1e-12, 5e-6);
}

#undef METRO_SAMPLER
#undef IMPO_SAMPLER
