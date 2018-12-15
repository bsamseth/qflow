#pragma once

#include <Eigen/Dense>
#include <cmath>
#include "definitions.hpp"

using Vector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
using VectorXr = Vector;

template<typename T>
inline Vector vector_from_sequence(const T& seq) {
    Vector vec(seq.size());
    for (auto it = seq.begin(); it != seq.end(); ++it) {
        vec[std::distance(seq.begin(), it)] = *it;
    }
    return vec;
}

template<typename T>
inline Real squaredNorm(const T& vec) {
    assert(vec.cols() == 1);
    return square(vec.array()).sum();
}
template<typename T>
inline Real norm(const T& vec) {
    return std::sqrt(squaredNorm(vec));
}

