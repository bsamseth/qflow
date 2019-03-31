#include "distance.hpp"

namespace Distance
{
namespace
{
bool                                               tracking_ = false;
Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> dirty_;
Eigen::Array<Real, Eigen::Dynamic, Eigen::Dynamic> dist_;
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

void stop_tracking()
{
    tracking_ = false;
}

Real probe(const System& system, int i, int j)
{
    if (tracking_)
    {
        assert(system.rows() == dirty_.rows() && system.rows() == dirty_.cols());
        if (dirty_(i, j))
        {
            dist_(i, j) = dist_(j, i) = norm(system.row(i) - system.row(j));
            dirty_(i, j) = dirty_(j, i) = false;
        }
        return dist_(i, j);
    }
    // Not tracking distances, fall back to just compute on demand.
    return norm(system.row(i) - system.row(j));
}

void invalidate_cache(int i)
{
    if (tracking_)
    {
        dirty_.row(i) = true;
        dirty_.col(i) = true;
    }
}

}  // namespace Distance
