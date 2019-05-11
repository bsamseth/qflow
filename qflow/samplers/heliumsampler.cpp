#include "heliumsampler.hpp"

#include <cmath>

HeliumSampler::HeliumSampler(const System& system,
                             Wavefunction& psi,
                             Real          step,
                             Real          box_size)
    : Sampler(system, psi, step), L(box_size), rho(system.rows() / (L * L * L))
{
    initialize_system();
}

void HeliumSampler::initialize_system()
{
    int    ncx       = 3;
    int    ncy       = 3;
    int    ncz       = 3;
    double basisx[4] = {0., 0.5, 0.0, 0.5};
    double basisy[4] = {0., 0.5, 0.5, 0.0};
    double basisz[4] = {0., 0.0, 0.5, 0.5};

    int    i, j, k, ll;
    double side, xd, yd, zd;
    int    index;

    side = std::pow(4. / rho, 1. / 3.);

    // printf("elementary cell side: %10.5f\n", side);

    index = 0;
    for (i = 0; i < ncx; i++)
    {
        for (j = 0; j < ncy; j++)
        {
            for (k = 0; k < ncz; k++)
            {
                for (ll = 0; ll < 4; ll++)
                {
                    _system_old(index, 0) = ((double) i + basisx[ll]) * side;
                    _system_old(index, 1) = ((double) j + basisy[ll]) * side;
                    _system_old(index, 2) = ((double) k + basisz[ll]) * side;
                    index++;
                }
            }
        }
    }

    for (j = 0; j < _system_old.rows(); j++)
    {
        xd                = _system_old(j, 0);
        _system_old(j, 0) = xd - L * std::rint(xd / L);
        yd                = _system_old(j, 1);
        _system_old(j, 1) = yd - L * std::rint(yd / L);
        zd                = _system_old(j, 2);
        _system_old(j, 2) = zd - L * std::rint(zd / L);
        // printf("%10.5f  %10.5f   %10.5f\n",
        //        _system_old(j, 0),
        //        _system_old(j, 1),
        //        _system_old(j, 2));
    }
    _system_new = _system_old;
    _psi_old    = (*_wavefunction)(_system_old);
}

void HeliumSampler::perturb_system()
{
    for (int d = 0; d < _system_new.cols(); ++d)
    {
        _system_new(_particle_to_move, d) += _step * centered(rand_gen);
        _system_new(_particle_to_move, d)
            -= L * std::rint(_system_new(_particle_to_move, d) / L);
    }
    _psi_new = (*_wavefunction)(_system_new);
}

Real HeliumSampler::acceptance_probability() const
{
    return square(_psi_new) / square(_psi_old);
}
