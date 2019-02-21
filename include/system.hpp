#pragma once

#include <Eigen/Dense>
#include "definitions.hpp"
#include "vector.hpp"

using System = Matrix;

inline Real distance(const System& system, int i, int j) {
    return norm(system.row(i) - system.row(j));
}
