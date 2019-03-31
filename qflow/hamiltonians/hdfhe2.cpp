#include "hdfhe2.hpp"

namespace
{
constexpr Real F(const Real r)
{
    if (r > HDFHE2::D * HDFHE2::r_m)
        return 1;

    return std::exp(-square(HDFHE2::D * HDFHE2::r_m / r - 1));
}
}  // namespace

Real HDFHE2::internal_potential(const System& system) const
{
    const int N     = system.rows();
    Real      total = 0;
    for (int i = 0; i < N - 1; ++i)
    {
        for (int j = i + 1; j < N; ++j)
        {
            const Real r     = distance(system, i, j);
            const Real rm_r  = HDFHE2::r_m / r;
            const Real rm_r2 = square(rm_r);
            const Real rm_r3 = rm_r2 * rm_r;

            total += HDFHE2::eps
                     * (HDFHE2::A * std::exp(-HDFHE2::alpha / rm_r)
                        - (HDFHE2::C_6 * square(rm_r3)
                           + HDFHE2::C_8 * square(square(rm_r2))
                           + HDFHE2::C_10 * square(rm_r2 * rm_r3))
                              * F(r));
        }
    }
    return total;
}
