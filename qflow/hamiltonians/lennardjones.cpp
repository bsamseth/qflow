#include "lennardjones.hpp"

#include "distance.hpp"

namespace
{
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

Real LennardJones::vlj_core(Real r2) const
{
    const Real l22 = square(0.5 * L);

    if (r2 > l22)
        return 0;

    if (r2 < square(0.3 * LennardJones::sigma))
        r2 = square(0.3 * LennardJones::sigma);

    Real r2i  = square(LennardJones::sigma) / r2;
    Real r6i  = r2i * r2i * r2i;
    Real r12i = r6i * r6i;

    return 4 * LennardJones::eps * (r12i - r6i) - truncation_potential;
}

LennardJones::LennardJones(Real box_size, Real h)
    : Hamiltonian(h, LennardJones::hbar2_per_m), L(box_size)
{
    // Compute the potential at a distance of half the simulation box.
    // Contributions from particles further away than that will be truncated, and
    // we account for this by shifting the potential for all smaller distances by
    // this amount.
    truncation_potential = 0;  // Important, truncation_potential is used in
                               // vlj_core, so zero before assignment.
    truncation_potential = vlj_core(square(0.5 * L) - 1e-10);
}

Real LennardJones::internal_potential(const System& system) const
{
    const int N = system.rows();

    Real total = 0;
    for (int i = 1; i < N; ++i)
    {
        for (int j = 0; j < i; ++j)
        {
            Real dx = system(i, 0) - system(j, 0);
            dx -= L * std::rint(dx / L);
            Real dy = system(i, 1) - system(j, 1);
            dy -= L * std::rint(dy / L);
            Real dz = system(i, 2) - system(j, 2);
            dz -= L * std::rint(dz / L);
            Real r2 = dx * dx + dy * dy + dz * dz;
            total += vlj_core(r2);
        }
    }

    const Real tail_correct = vtail(L, N);
    const Real bulk_correct = 0.5 * truncation_potential * (PI * N / 6.0 - 1);
    return total + (tail_correct + bulk_correct) * N;
}
