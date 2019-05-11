#include "lennardjones.hpp"

#include "distance.hpp"

namespace
{
Real lj_core(Real r, Real L)
{
    // Pairs that exceed half the simulation box get zero contribution.
    if (2 * r > L)
        return 0;

    Real r_fixed = r < LennardJones::r_core ? LennardJones::r_core : r;

    Real sigma_r   = LennardJones::sigma / r_fixed;
    Real sigma_r_6 = square(sigma_r * sigma_r * sigma_r);
    return 4 * LennardJones::eps * (sigma_r_6 * sigma_r_6 - sigma_r_6);
}

Real vtail(const Real L, const int n_particles)
{
    Real Li     = 1 / L;
    Real aux    = 2. * LennardJones::sigma * Li;
    Real aux3   = aux * aux * aux;
    Real aux9   = aux3 * aux3 * aux3;
    Real sigma3 = LennardJones::sigma * LennardJones::sigma * LennardJones::sigma;
    Real rho    = n_particles * Li * Li * Li;
    return 8.0 / 9.0 * PI * LennardJones::eps * rho * sigma3 * (aux9 - 3 * aux3);
}

}  // namespace

LennardJones::LennardJones(Real h) : Hamiltonian(h, LennardJones::hbar2_per_m) {}

Real LennardJones::internal_potential(const System& system) const
{
    const int  N            = system.rows();
    const Real L            = Distance::get_simulation_box_size();
    const Real tail_correct = vtail(L, N);

    Real total = 0;
    for (int i = 0; i < N - 1; ++i)
    {
        for (int j = i + 1; j < N; ++j)
        {
            total += lj_core(Distance::probe(system, i, j), L);
        }
    }
    return total;
}
