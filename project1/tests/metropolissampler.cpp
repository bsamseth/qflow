#include <gtest/gtest.h>

#include "definitions.hpp"
#include "system.hpp"
#include "simplegaussian.hpp"
#include "harmonicoscillatorhamiltonian.hpp"
#include "metropolissampler.hpp"

TEST(MetropolisSampler, intergrationTest) {
    const Real gamma = 2.82843;
    const Real alpha = 0.5;
    const int runs = 10000;
    const System init_system (10, 3);
    const SimpleGaussian psi(alpha, gamma);
    const HarmonicOscillatorHamiltonian H(gamma);

    MetropolisSampler sampler (init_system, psi, 1.0);

    Real E_analytic = 0;
    Real E_numeric = 0;
    for (int run = 0; run < runs; ++run) {
        const System &state = sampler.next_configuration();

        E_analytic += H.local_energy(state, psi);
        E_numeric += H.local_energy_numeric(state, psi);
    }

    E_analytic /= runs;
    E_numeric /= runs;

    EXPECT_NEAR(H.gross_pitaevskii_energy(init_system, psi), E_analytic, 1e-12);
    EXPECT_NEAR(H.gross_pitaevskii_energy(init_system, psi), E_numeric, 5e-6);
}
