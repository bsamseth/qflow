#include "jastrowmcmillian.hpp"

#include "distance.hpp"

#include <cmath>

JastrowMcMillian::JastrowMcMillian(int n, Real beta, Real box_size)
    : Wavefunction(vector_from_sequence({beta})), n_(n), L(box_size)
{
    assert(_parameters[0] == beta);
    assert(_parameters.size() == 1);
}

Real JastrowMcMillian::operator()(const System& system)
{
    const Real beta = _parameters[0];
    const int  N    = system.rows();
    Real       res  = 0;
    Real       l22  = square(0.5 * L);
    for (int i = 0; i < N - 1; ++i)
    {
        for (int j = i + 1; j < N; ++j)
        {
            Real dx = system(i, 0) - system(j, 0);
            dx -= L * std::rint(dx / L);
            Real dy = system(i, 1) - system(j, 1);
            dy -= L * std::rint(dy / L);
            Real dz = system(i, 2) - system(j, 2);
            dz -= L * std::rint(dz / L);

            Real rr = dx * dx + dy * dy + dz * dz;

            if (rr > l22)
                continue;

            if (rr < square(0.3 * 2.556))
                rr = square(0.3 * 2.556);

            Real ri = beta / std::sqrt(rr);
            res += 0.5 * std::pow(ri, n_);
        }
    }
    return std::exp(-res);
}

RowVector JastrowMcMillian::gradient(const System& system)
{
    /*
      const Real beta = _parameters[0];
      RowVector  grad = RowVector::Zero(1);
      const int  N    = system.rows();
      for (int i = 0; i < N - 1; ++i)
      {
          for (int j = i + 1; j < N; ++j)
          {
              const Real r_ij = Distance::probe(system, i, j);
              grad[0] += std::pow(1 / r_ij, n_);
          }
      }
      grad[0] *= -0.5 * n_ * std::pow(beta, n_ - 1);
      return grad;
    */
    return RowVector::Zero(1);
}

/*
Real JastrowMcMillian::drift_force(const System& system, int k, int d)
{
    const Real beta = _parameters[0];
    const int  N    = system.rows();
    Real       res  = 0;
    for (int i = 0; i < N; ++i)
    {
        if (i != k)
        {
            const Real r_ik = Distance::probe(system, i, k);
            res += std::pow(beta / r_ik, n_) * (system(k, d) - system(i, d))
                   / (r_ik * r_ik);
        }
    }
    return 2 * 0.5 * n_ * res;
}
*/

Real JastrowMcMillian::laplacian(const System& system)
{
    const Real beta = _parameters[0];
    const int  N    = system.rows();
    const int  D    = system.cols();
    const Real l22  = square(0.5 * L);

    Real   d2psi = 0;
    System dpsi  = System::Zero(N, D);

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

            if (r2 < l22)
            {
                if (r2 < square(0.3 * 2.556))
                    r2 = square(0.3 * 2.556);

                Real r2i = beta * beta / r2;
                Real ri  = 1. / (beta * std::sqrt(r2));
                Real r6i = r2i * r2i * r2i;

                Real v = -5 * r6i * ri;

                Real dux = v * dx;
                Real duy = v * dy;
                Real duz = v * dz;

                dpsi(i, 0) += dux;
                dpsi(i, 1) += duy;
                dpsi(i, 2) += duz;
                dpsi(j, 0) -= dux;
                dpsi(j, 1) -= duy;
                dpsi(j, 2) -= duz;

                d2psi += 20. * r6i * ri;
            }
        }
    }
    Real tpb = 0;
    for (int i = 0; i < N; ++i)
    {
        tpb += 0.25 * square(dpsi(i, 0));
        tpb += 0.25 * square(dpsi(i, 1));
        tpb += 0.25 * square(dpsi(i, 2));
    }

    return tpb - d2psi;
}
