#include <cstdlib>
#include <iostream>
#include <fstream>
#include <armadillo>
#include <vector>
#include <string>
#include <chrono>

#include "vmcsolver.hpp"

using namespace std;
using namespace VMC;

int main(int argc, char *argv[])
{
    if (argc < 6) {
        printf("Usage: ./main.x n_cycles alpha_min alpha_max alpha_step [filename]\n");
        return 0;
    }

    // Read in parameters.
    const int n_cycles      = atoi(argv[1]);
    const double alpha_min  = atof(argv[2]);
    const double alpha_max  = atof(argv[3]);
    const double alpha_n    = atof(argv[4]);

    // Set output stream to cout or file, depending on valid filename input.
    std::ofstream out_file (argv[5]);
    if (!out_file.is_open()) {
        printf("Could not open file '%s'\n", argv[5]);
        return 1;
    }


    // Setup configuration.

    VMCConfiguration config;
    config.ho_type = HOType::SYMMETRIC;
    config.interaction = InteractionType::OFF;
    config.omega_ho = 1;
    config.omega_z = 1;
    config.a = 0.0043;
    config.h = 0.001;
    config.h2 = 1 / (config.h * config.h);
    config.step_length = 1;


    // Run vmc for all versions wanted in b)

    vector<Dimensions> dimensions { DIM_1, DIM_2, DIM_3 };
    vector<int> particles { 1, 10, 100, 500 };
    vector<AnalyticAcceleration> analytic_on_off
            { AnalyticAcceleration::ON, AnalyticAcceleration::OFF };

    for (auto dims : dimensions) {
        config.dims = dims;
        for (auto n_particles : particles) {
            config.n_particles = n_particles;
            for (auto analytic : analytic_on_off) {
                config.acceleration = analytic;

                if (analytic == AnalyticAcceleration::OFF and n_particles > 100) continue;

                auto start_time = chrono::high_resolution_clock::now();

                VMCSolver vmc(config);

                Results result = vmc.vmc(n_cycles,
                                         out_file,
                                         alpha_min,
                                         alpha_max,
                                         alpha_n);

                auto end_time = chrono::high_resolution_clock::now();
                int milli_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();

                printf("Dims = %d, N = %3d, Analytic = %3s: ",
                       config.dims, config.n_particles,
                       config.acceleration == AnalyticAcceleration::ON ? "ON" : "OFF");
                cout << result;
                printf(" : in time %d ms\n", milli_time);
                cout << flush;
            }
        }
    }

    cout << "Done!" << endl;

    return 0;
}
