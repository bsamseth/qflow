#include <gtest/gtest.h>

#include "definitions.hpp"
#include "system.hpp"
#include "simplegaussian.hpp"
#include "harmonicoscillatorhamiltonian.hpp"
#include "metropolissampler.hpp"
#include "optimizer.hpp"

TEST(GradientDecent, findsOptmalForNonInteracting) {

    System init_system(10, 3);
    SimpleGaussian psi({-1, 1});
    HarmonicOscillatorHamiltonian H(1);
    MetropolisSampler sampler(init_system, psi, 1);

    Real learning_rate = 0.01;
    int n_cycles = 10000;
    int max_iterations = 1e4;
    Real minimum_gradient = 1e-6;

    for (Real initial_guess = 0.3; initial_guess < 1; initial_guess += 0.4) {



        Real alpha_optimal = Optimizer::gradient_decent_optimizer(psi,
                                                                  H,
                                                                  sampler,
                                                                  initial_guess,
                                                                  learning_rate,
                                                                  n_cycles,
                                                                  max_iterations,
                                                                  minimum_gradient,
                                                                  false);
        ASSERT_NEAR(0.5, alpha_optimal, 5e-8);
    }
}


