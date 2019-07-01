#include "heliumsampler.hpp"

#include <cmath>
#include <exception>

namespace
{
struct GridPoints
{
    int x, y, z;
};

GridPoints find_cell_dimensions(int n)
{
  int x = std::pow(n, 1.0 / 3) + 1;
  while (n % x != 0)
    x--;

  int y = std::sqrt(n / x) + 1;
  while ((n / x) % y != 0)
    y--;

  int z = (n / x) / y;
  return {x, y, z};
}

}  // namespace

HeliumSampler::HeliumSampler(const System& system,
                             Wavefunction& psi,
                             Real          step,
                             Real          box_size)
    : Sampler(system, psi, step), L(box_size)
{
    initialize_system();
}

void HeliumSampler::initialize_system()
{
    if (_system_old.rows() % 4 != 0)
        throw std::invalid_argument("System #rows must be divisible by four.");

    auto [ncx, ncy, ncz] = find_cell_dimensions(_system_old.rows() / 4);
    Real basisx[4]       = {0., 0.5, 0.0, 0.5};
    Real basisy[4]       = {0., 0.5, 0.5, 0.0};
    Real basisz[4]       = {0., 0.0, 0.5, 0.5};

    const int  N    = _system_old.rows();
    const Real side = std::pow(4. * L * L * L / N, 1. / 3.);

    int index = 0;
    for (int i = 0; i < ncx; i++)
        for (int j = 0; j < ncy; j++)
            for (int k = 0; k < ncz; k++)
                for (int ll = 0; ll < 4; ll++)
                {
                    _system_old(index, 0) = ((double) i + basisx[ll]) * side;
                    _system_old(index, 1) = ((double) j + basisy[ll]) * side;
                    _system_old(index, 2) = ((double) k + basisz[ll]) * side;
                    index++;
                }

    // Make sure all point are mapped into the simulation box.
    for (int j = 0; j < _system_old.rows(); ++j)
        for (int k = 0; k < _system_old.cols(); ++k)
            _system_old(j, k) -= L * std::rint(_system_old(j, k) / L);

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
