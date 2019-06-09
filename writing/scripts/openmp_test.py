import numpy as np
import matplotlib.pyplot as plt
import matplotlib2tikz

# Source: https://gist.github.com/bbengfort/bf62e3487b9732daebd5
##include <omp.h>
##include <stdio.h>
##include <stdlib.h>
##define MAX_THREADS 64

# static long steps = 1000000000;
# double step;
# int main (int argc, const char *argv[]) {
#    int i,j;
#    double x;
#    double pi, sum = 0.0;
#    double start, delta;
#    step = 1.0/(double) steps;
#    // Compute parallel compute times for 1-MAX_THREADS
#    for (j=1; j<= MAX_THREADS; j++) {
#        printf(" running on %d threads: ", j);
#        // This is the beginning of a single PI computation
#        omp_set_num_threads(j);
#        sum = 0.0;
#        double start = omp_get_wtime();
#        #pragma omp parallel for reduction(+:sum) private(x)
#        for (i=0; i < steps; i++) {
#            x = (i+0.5)*step;
#            sum += 4.0 / (1.0+x*x);
#        }
#        // Out of the parallel region, finialize computation
#        pi = step * sum;
#        delta = omp_get_wtime() - start;
#        printf("PI = %.16g computed in %.4g seconds\n", pi, delta);
#    }
# }

# Following data produced by running the above script on a login node at abel.
times = eval(  # Eval to hide from black formatter
    "np.array([3.448 ,1.709 ,1.139 ,0.8475 ,0.6914 ,0.5786 ,.5079 ,0.4407 ,0.3954 , 0.3574 , 0.3254 , 0.2976 , 0.2829 , 0.2686 , 0.2451 , 0.2298 , 0.3369 , 0.3896 ,0.3675 , 0.356 , 0.3403 ,0.2996 ,0.2834 , 0.3015 ,0.2905 , 0.2794 , 0.2693 , 0.2647 , 0.2525 , 0.2459 , 0.2388 , 0.2754 , 0.3056 , 0.2733])"
)
cpus = np.arange(1, len(times) + 1)

plt.plot(cpus, times[0] / times, label="Actual")
plt.plot(cpus, cpus, "--", label="Ideal speedup")
plt.legend()
plt.xlabel("Threads")
plt.ylabel("Iterations per second")

matplotlib2tikz.save(__file__ + ".tex")
plt.show()
