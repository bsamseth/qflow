#pragma once

#include "definitions.hpp"
#include "vector.hpp"

#include <Eigen/Dense>

using System = Matrix;

inline Real distance(const System& system, int i, int j)
{
    return norm(system.row(i) - system.row(j));
}
