#include "hfdhe2.hpp"

#include "distance.hpp"

namespace
{
constexpr Real F(const Real r)
{
    if (r > HFDHE2::D * HFDHE2::r_m)
        return 1;

    return std::exp(-square(HFDHE2::D * HFDHE2::r_m / r - 1));
}
}  // namespace

Real HFDHE2::internal_potential(const System& system) const
{
    const int N     = system.rows();
    Real      total = 0;
    for (int i = 0; i < N - 1; ++i)
    {
        for (int j = i + 1; j < N; ++j)
        {
            const Real r     = Distance::probe(system, i, j);
            const Real rm_r  = HFDHE2::r_m / r;
            const Real rm_r2 = square(rm_r);
            const Real rm_r3 = rm_r2 * rm_r;

            total += HFDHE2::eps
                     * (HFDHE2::A * std::exp(-HFDHE2::alpha / rm_r)
                        - (HFDHE2::C_6 * square(rm_r3)
                           + HFDHE2::C_8 * square(square(rm_r2))
                           + HFDHE2::C_10 * square(rm_r2 * rm_r3))
                              * F(r));
        }
    }
    return total;
}
