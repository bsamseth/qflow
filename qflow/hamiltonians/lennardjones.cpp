#include "lennardjones.hpp"

#include "distance.hpp"

Real LennardJones::internal_potential(const System& system) const
{
    const int N     = system.rows();
    Real      total = 0;
    for (int i = 0; i < N - 1; ++i)
    {
        for (int j = i + 1; j < N; ++j)
        {
            const Real sigma_r   = LennardJones::sigma / Distance::probe(system, i, j);
            const Real sigma_r_6 = square(sigma_r * sigma_r * sigma_r);

            total += sigma_r_6 * sigma_r_6 - sigma_r_6;
        }
    }
    return 4 * LennardJones::eps * total;
}
