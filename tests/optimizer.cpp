#include "optimizer.hpp"

#include "definitions.hpp"
#include "harmonicoscillator.hpp"
#include "importancesampler.hpp"
#include "simplegaussian.hpp"
#include "system.hpp"

#include <gtest/gtest.h>

TEST(GradientDecent, findsOptmalForNonInteracting)
{
    System                        init_system = System::Zero(10, 3);
    SimpleGaussian                psi(0.3, 1);
    HarmonicOscillator H(1);
    ImportanceSampler             sampler(init_system, psi, 0.5);

    AdamOptimizer adam(psi.get_parameters().size(), 0.01);

    for (Real initial_guess = 0.3; initial_guess < 1; initial_guess += 0.4)
    {
        H.optimize_wavefunction(psi, sampler, 2000, 100, adam, 0, false);

        ASSERT_NEAR(0.5, psi.get_parameters()[0], 5e-4);
    }
}
