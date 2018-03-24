#include <cstdlib>
#include <fstream>
#include <iostream>

#include "vmc.hpp"

#define intarg(i) (atoi(argv[i]))
#define floatarg(i) (atof(argv[i]))

void parse_arguments(int argc, char **argv, Wavefunction **psi, Hamiltonian **H, Sampler **sampler, Calculator **calc, int *cycles) {
    if (argc - 1 < 11) {
        printf("Usage: ./main.x analytic[OFF=0,ON=1] sampler[Metro=0,Imp=1] dim n_bosons n_cycles alpha beta gamma a step filename \n");
        std::exit(0);
    }

    bool analytic = intarg(1) != 0;
    int sampler_type = intarg(2);
    int dimensions = intarg(3);
    int bosons = intarg(4);
    *cycles = intarg(5);
    Real alpha = floatarg(6);
    Real beta = floatarg(7);
    Real gamma = floatarg(8);
    Real a = floatarg(9);
    Real step = floatarg(10);
    std::string filename = argv[11];

    System *system = new System(bosons, dimensions);

    if (a != 0) {
        *psi = new InteractingWavefunction(alpha, beta, a);
        *H = new InteractingHamiltonian(gamma, a);
    } else {
        *psi = new SimpleGaussian(alpha, beta);
        *H = new HarmonicOscillatorHamiltonian(gamma);
    }

    if (sampler_type == 0) {
        *sampler = new MetropolisSampler(*system, **psi, step);
    } else {
        *sampler = new ImportanceSampler(*system, **psi, step);
    }

    *calc = new EnergyCalculator(**psi, **H, **sampler, filename, analytic);
}

#undef intarg
#undef floatarg
