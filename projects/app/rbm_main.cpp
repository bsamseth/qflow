#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>

#include "rbm.hpp"

using std::cout;
using std::endl;

int main(int argc, char **argv) {

    if (argc < 6) {
        printf("Usage: %s N iterations samples omega gamma\n", argv[0]);
        return 0;
    }
    const int N = atoi(argv[1]);
    const long iterations = atoi(argv[2]);
    const long samples = atoi(argv[3]);
    const Real omega = atof(argv[4]);
    const Real gamma = atof(argv[5]);

    rand_gen.seed(1234);

    System init_system (2, 2);
    RBMWavefunction rbm (4, N, 1);
    ImportanceSampler sampler (init_system, rbm, 0.1);
    RBMInteractingHamiltonian H(omega);

    std::cout << rbm.get_parameters() << std::endl;
    Real E = 0;
    for (long i = 0; i < (long) 1e6; ++i)
        E += H.local_energy(sampler.next_configuration(), rbm);
    std::cout << "Before training: <E> = " << E / 1e6 << std::endl;

    AdamOptimizer optimizer(rbm.get_parameters().size());
    rbm.train(H, sampler, iterations, samples, optimizer, gamma, true);
    printf("AR = %g\n", sampler.get_acceptance_rate());
    std::cout << rbm.get_parameters() << std::endl;

    E = 0;
    for (long i = 0; i < (long) 1e6; ++i)
        E += H.local_energy(sampler.next_configuration(), rbm);
    std::cout << "After training: <E> = " << E / 1e6 << std::endl;
}
