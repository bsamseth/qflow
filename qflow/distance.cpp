#include "distance.hpp"

#include <stdexcept>

namespace Distance
{
namespace
{
bool                                               tracking_ = false;
Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> dirty_;
Eigen::Array<Real, Eigen::Dynamic, Eigen::Dynamic> dist_;
Real pbc_size = -1;  // Negative indicates no size set.
}  // namespace

void start_tracking(const System& system)
{
    if (!tracking_ || (system.rows() != dist_.rows()))
    {
        dirty_ = Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>::Ones(
            system.rows(), system.rows());
        dist_ = Eigen::Array<Real, Eigen::Dynamic, Eigen::Dynamic>::Zero(system.rows(),
                                                                         system.rows());
        tracking_ = true;
    }
}

void start_tracking(const System& system, Real L)
{
    if (!tracking_ || (system.rows() != dist_.rows()) || pbc_size != L)
    {
        set_simulation_box_size(L);
        tracking_ = false;  // Make sure we refresh cache also if only size has changed.
        start_tracking(system);
    }
}

void stop_tracking()
{
    tracking_ = false;
    disable_simulation_box();
}

void set_simulation_box_size(Real size)
{
    if (size <= 0)
        throw std::invalid_argument("Simulation box must have size > 0.");
    pbc_size = size;
}

void disable_simulation_box()
{
    pbc_size = -1;
}

Real get_simulation_box_size()
{
    return pbc_size;
}

void invalidate_cache(int i)
{
    if (tracking_)
    {
        dirty_.row(i) = true;
        dirty_.col(i) = true;
    }
}

Real probe(const System& system, int i, int j)
{
    auto distance_metric
        = pbc_size <= 0 ? [](const System& system, int i, int j) -> Real {
        return norm(system.row(i) - system.row(j));
    }
    : [](const System& system, int i, int j) -> Real {
          auto diff = (system.row(i) - system.row(j)).array();
          return norm(diff - Eigen::round(diff / pbc_size) * pbc_size);
      };
    if (tracking_)
    {
        assert(system.rows() == dirty_.rows() && system.rows() == dirty_.cols());
        if (dirty_(i, j))
        {
            dist_(i, j) = dist_(j, i) = distance_metric(system, i, j);
            dirty_(i, j) = dirty_(j, i) = false;
        }
        return dist_(i, j);
    }
    // Not tracking distances, fall back to just compute on demand.
    return distance_metric(system, i, j);
}
}  // namespace Distance
