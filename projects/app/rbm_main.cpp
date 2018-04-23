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

    System init_system (1, 1);
    RBMWavefunction rbm (1, 2);
    MetropolisSampler sampler (init_system, rbm, 0.5);
    RBMHarmonicOscillatorHamiltonian H;

    rbm.train(H, sampler, 10000, 1000000, 0.01);

}
