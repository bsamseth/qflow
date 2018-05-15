#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>

#include "rbm.hpp"

using std::cout;
using std::endl;

int main() {

    System init_system (2, 2);
    RBMWavefunction rbm (4, 10, 1);
    ImportanceSampler sampler (init_system, rbm, 0.5);
    RBMInteractingHamiltonian H;


    std::cout << rbm.get_parameters() << std::endl;

    AdamOptimizer optimizer(rbm.get_parameters().size());
    rbm.train(H, sampler, 1000, 1e3, optimizer, 0.0, true);
    printf("AR = %g\n", sampler.get_acceptance_rate());
    std::cout << rbm.get_parameters() << std::endl;

    Real E = 0;
    for (long i = 0; i < (long) 1e6; ++i)
        E += H.local_energy(sampler.next_configuration(), rbm);
    std::cout << "<E> = " << E / 1e6 << std::endl;
}
