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

    Real initial_guess;
    Real learning_rate;
    int sample_points_per_iteration;
    int max_iterations = 10000;
    Real minimum_gradient;

    parse_arguments_optimize(argc, argv, &psi, &H, &sampler, &initial_guess,
                             &learning_rate, &sample_points_per_iteration,
                             &minimum_gradient);

    cout << *psi << endl;
    cout << *H << endl;
    cout << *sampler << endl;
    cout << sampler->get_current_system() << endl;
    cout << "Alpha guess: " << initial_guess << ", "
         << "Learning rate: " << learning_rate << ", "
         << "Cycles per iteration: " << sample_points_per_iteration << ", "
         << "Max iterations: " << max_iterations << ", "
         << "Minimum gradient: " << minimum_gradient << endl;


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
