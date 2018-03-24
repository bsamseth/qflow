#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>

#include "vmc.hpp"

using std::cout;
using std::endl;

int main(int argc, char *argv[]) {

    Wavefunction *psi = nullptr;
    Hamiltonian *H = nullptr;
    Sampler *sampler = nullptr;
    Calculator *calc = nullptr;
    int cycles = 0;

    parse_arguments(argc, argv, &psi, &H, &sampler, &calc, &cycles);

    auto start_time = std::chrono::high_resolution_clock::now();
    calc->calculate(cycles);
    auto end_time = std::chrono::high_resolution_clock::now();
    long long micro_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

    cout.precision(10);
    cout << std::scientific << sampler->get_acceptance_rate() << ", " << micro_time / (double) 1e6 << "\n";

    return 0;
}
