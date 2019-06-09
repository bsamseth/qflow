#include <iostream>
#include <omp.h>

#define MAX_THREADS 8

int main(int argc, const char* argv[])
{
    constexpr long steps = 1000000000;

    // Compute parallel compute times for 1-MAX_THREADS
    for (int j = 1; j <= MAX_THREADS; j++)
    {
        std::cout << "running on " << j << " threads: ";

        omp_set_num_threads(j);

        double       sum   = 0.0;
        const double start = omp_get_wtime();

#pragma omp parallel for reduction(+ : sum)
        for (int i = 0; i < steps; i++)
        {
            const double x = (i + 0.5) / steps;
            sum += 4.0 / (1.0 + x * x);
        }

        const double pi    = sum / steps;
        const double delta = omp_get_wtime() - start;
        std::cout << "PI = " << pi << " computed in " << delta << " seconds"
                  << std::endl;
    }
}
