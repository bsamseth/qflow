#include <iostream>
#include <armadillo>

#include "vmcsolver.h"
using namespace std;

int main(int argc, char *argv[])
{
    constexpr int n_particles = 3;
    constexpr VMC::Dimensions dims = VMC::DIM_3;
    constexpr VMC::HOType ho_type = VMC::HOType::SYMMETRIC;
    constexpr VMC::InteractionType interaction = VMC::InteractionType::OFF;
    constexpr VMC::AnalyticAcceleration analytic = VMC::AnalyticAcceleration::ON;
    constexpr int mass = 1;
    constexpr int omega_ho = 1;
    constexpr int n_cycles = 10;

    VMC::VMCSolver<n_particles, dims, ho_type, interaction, analytic> vmc(mass, omega_ho);
    vmc.run(n_cycles);

    return 0;
}
