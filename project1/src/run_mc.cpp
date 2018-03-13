#include <cstdlib>
#include <iostream>
#include <fstream>
#include <armadillo>
#include <vector>
#include <string>
#include <chrono>

#include "vmcsolver.hpp"
#include "vmcimportancesolver.hpp"

using namespace std;
using namespace VMC;

#define intarg(i) (atoi(argv[i]))
#define floatarg(i) (atof(argv[i]))

int main(int argc, char *argv[])
{
    if (argc - 1 < 14) {
        printf("Usage: ./main.x analytic[0=OFF,1=ON] importance[0=OFF,1=ON] ho_type[0=sym,1=elip] dims n_particles n_cycles "
               "alpha beta time_step step_length omega_ho omega_z a h filename\n");
        return 0;
    }

    // Default configs.
    VMCConfiguration config;
    config.acceleration = intarg(1) == 0 ? AnalyticAcceleration::OFF : AnalyticAcceleration::ON;
    const bool use_importance = intarg(2);
    config.ho_type = intarg(3) == 0 ? HOType::SYMMETRIC : HOType::ELLIPTICAL;
    config.dims = Dimensions(intarg(4));
    config.n_particles = intarg(5);
    const int n_cycles = intarg(6);
    const double alpha = floatarg(7);
    const double beta  = floatarg(8);
    config.time_step   = floatarg(9);
    config.step_length = floatarg(10);
    config.omega_ho    = floatarg(11);
    config.omega_z     = floatarg(12);
    config.a           = floatarg(13);
    config.interaction = config.a == 0 ? InteractionType::OFF : InteractionType::ON;
    config.h           = floatarg(14);
    config.h2          = 1 / (config.h * config.h);


    VMCSolver *vmc;
    if (use_importance)
        vmc = new VMCImportanceSolver(config);
    else
        vmc = new VMCSolver(config);

    ofstream out (argv[15], ios::out | ios::binary);
    if (!out.is_open()) {
        cout << "Error opening file: " << argv[15] << endl;
        return 1;
    }

    auto start_time = chrono::high_resolution_clock::now();

    Results result = vmc->run_MC(n_cycles, &out, alpha, beta);

    auto end_time = chrono::high_resolution_clock::now();
    long long micro_time = chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();

    cout.precision(16);
    cout << scientific << result << ", " << micro_time / (double) 1e6 << "\n";

    out.close();

    return 0;
}
