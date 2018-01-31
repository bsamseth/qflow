#include <cstdlib>
#include <iostream>
#include <fstream>
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
    constexpr int omega_ho = 1;

    if (argc < 5) {
        printf("Usage: ./main.x n_cycles alpha_min alpha_max alpha_step [filename]\n");
        return 0;
    }

    // Read in parameters.
    const int n_cycles      = atoi(argv[1]);
    const double alpha_min  = atof(argv[2]);
    const double alpha_max  = atof(argv[3]);
    const double alpha_step = atof(argv[4]);

    // Set output stream to cout or file, depending on valid filename input.
    streambuf * buf;
    std::ofstream out_file;
    if(argc == 6) {
        const char *filename = argv[5];
        out_file.open(filename);
        if (!out_file.is_open()) {
            printf("Could not open file '%s'\n", filename);
            return 1;
        }
        buf = out_file.rdbuf();
    } else {
        buf = std::cout.rdbuf();
    }
    std::ostream out(buf);


    // Initialize the solver, and run VMC. Output directed to 'out'.
    VMC::VMCSolver<n_particles, dims, ho_type, interaction, analytic> vmc(omega_ho);
    VMC::Results best = vmc.vmc(n_cycles, out, alpha_min, alpha_max, alpha_step);
    printf("Var(alpha = %g) = %g\n", best.alpha, best.variance);
    return 0;
}
