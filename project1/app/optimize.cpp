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

    cout << *psi << endl;
    cout << *H << endl;
    cout << *sampler << endl;
    cout << sampler->get_current_system() << endl;

    Real initial_guess = 0.45;
    Real learning_rate = 0.1;
    int sample_points_per_iteration = 1000000;
    int max_iterations = 1000;
    Real minimum_gradient = 0.0000001;

    Real alpha_optimal = Optimizer::gradient_decent_optimizer(*psi,
                                                              *H,
                                                              *sampler,
                                                              initial_guess,
                                                              learning_rate,
                                                              sample_points_per_iteration,
                                                              max_iterations,
                                                              minimum_gradient);

    printf("Optimal alpha = %.10f\n", alpha_optimal);

    return 0;
}
