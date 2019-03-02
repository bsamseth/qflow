#pragma once

#include "definitions.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <initializer_list>

inline RowVector vector_from_sequence(std::initializer_list<Real> seq)
{
    RowVector vec(seq.size());
    for (auto it = seq.begin(); it != seq.end(); ++it)
    {
        vec[std::distance(seq.begin(), it)] = *it;
    }
    return vec;
}

template <typename T>
inline Real squaredNorm(const T& vec)
{
    assert(vec.rows() == 1);
    return square(vec.array()).sum();
}
template <typename T>
inline Real norm(const T& vec)
{
    return std::sqrt(squaredNorm(vec));
}
