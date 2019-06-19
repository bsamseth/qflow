#include "jastrowmcmillian.hpp"

#include "distance.hpp"
#include "lennardjones.hpp"

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
    for (int i = 0; i < N - 1; ++i)
    {
        for (int j = i + 1; j < N; ++j)
        {
            auto diff = (system.row(i) - system.row(j)).array();
            Real r    = norm(diff - Eigen::round(diff / L) * L);

            if (r > 0.5 * L)
                continue;

            // Avoid overflow by fixing a lower bound for the distance. The
            // likelihood of being here is very small, so the effect should be
            // minimal.
            r = std::max(LennardJones::r_core, r);

            res += std::pow(beta / r, n_);
        }
    }
    return std::exp(-0.5 * res);
}

RowVector JastrowMcMillian::gradient(const System& system)
{
    const Real beta = _parameters[0];
    RowVector  grad = RowVector::Zero(1);
    const int  N    = system.rows();
    for (int i = 0; i < N - 1; ++i)
    {
        for (int j = i + 1; j < N; ++j)
        {
            auto diff           = (system.row(i) - system.row(j)).array();
            auto diff_corrected = diff - Eigen::round(diff / L) * L;
            Real r              = norm(diff_corrected);

            if (r > 0.5 * L)
              continue;

            r = std::max(LennardJones::r_core, r);
            grad[0] += std::pow(1 / r, n_);
        }
    }
    grad[0] *= -0.5 * n_ * std::pow(beta, n_ - 1);
    return grad;
}

Real JastrowMcMillian::drift_force(const System& system, int k, int d)
{
    const Real beta = _parameters[0];
    const int  N    = system.rows();
    Real       res  = 0;
    for (int i = 0; i < N; ++i)
    {
        if (i != k)
        {
            auto diff           = (system.row(k) - system.row(i)).array();
            auto diff_corrected = diff - Eigen::round(diff / L) * L;
            Real r              = norm(diff_corrected);

            if (r > 0.5 * L)
              continue;

            r = std::max(LennardJones::r_core, r);
            res += std::pow(beta / r, n_) * diff_corrected[d] / (r * r);
        }
    }
    return 2 * 0.5 * n_ * res;
}

Real JastrowMcMillian::laplacian(const System& system)
{
    const Real beta = _parameters[0];
    const int  N    = system.rows();
    const int  D    = system.cols();

    Real   d2psi = 0;
    System dpsi  = System::Zero(N, D);

    for (int i = 1; i < N; ++i)
    {
        for (int j = 0; j < i; ++j)
        {
            auto diff           = (system.row(i) - system.row(j)).array();
            auto diff_corrected = diff - Eigen::round(diff / L) * L;
            Real r              = norm(diff_corrected);

            if (r > 0.5 * L)
              continue;

            r        = std::max(LennardJones::r_core, r);
            Real r2i = beta * beta / square(r);
            Real ri  = 1. / (beta * r);
            Real r6i = r2i * r2i * r2i;

            Real v = -5 * r6i * ri;

            RowVector du = v * diff_corrected;
            dpsi.row(i) += du;
            dpsi.row(j) -= du;

            d2psi += 20. * r6i * ri;
        }
    }

    return 0.25 * dpsi.squaredNorm() - d2psi;
}
