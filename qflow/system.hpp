#pragma once

#include "definitions.hpp"
#include "vector.hpp"

#include <Eigen/Dense>

using System = Matrix;

namespace Distance
{
void init(const System&);
Real probe(const System&, int i, int j);
void invalidate_cache(int i);

// The plain, non-memoized euclidian norm.
// Fall back to this if not initialized, or when cache is invalidated.
inline Real distance(const System& system, int i, int j)
{
    return norm(system.row(i) - system.row(j));
}

}  // namespace Distance
