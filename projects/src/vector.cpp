#include <vector>
#include <cassert>
#include <iostream>
#include "definitions.hpp"
#include "vector.hpp"

Vector::Vector(int dimensions) : _pos(dimensions, 0.0) { }

/*
 * Allowed for simplicity in creating test instances.
 */
Vector::Vector(const std::vector<Real> &pos) : _pos(pos) { }

Real operator* (const Vector &lhs, const Vector &rhs) {
    assert(lhs.get_dimensions() == rhs.get_dimensions());
    Real prod = 0;
    for (int i = 0; i < lhs.get_dimensions(); i++) {
        prod += lhs[i] * rhs[i];
    }
    return prod;
}
Vector& operator+= (Vector &lhs, const Vector &rhs) {
    assert(lhs.get_dimensions() == rhs.get_dimensions());
    for (int i = 0; i < lhs.get_dimensions(); i++) {
        lhs[i] += rhs[i];
    }
    return lhs;
}
Vector& operator-= (Vector &lhs, const Vector &rhs) {
    assert(lhs.get_dimensions() == rhs.get_dimensions());
    for (int i = 0; i < lhs.get_dimensions(); i++) {
        lhs[i] -= rhs[i];
    }
    return lhs;
}
Vector& operator+= (Vector &lhs, Real rhs) {
    for (int i = 0; i < lhs.get_dimensions(); i++) {
        lhs[i] += rhs;
    }
    return lhs;
}
Vector& operator*= (Vector &lhs, Real rhs) {
    for (int i = 0; i < lhs.get_dimensions(); i++) {
        lhs[i] *= rhs;
    }
    return lhs;
}

/*
 * All remaining implementations are made using the above, for simplicty,
 * code maintainability and bug-prevention. This might cause a slight
 * efficiency hit, but the compiler should be able to minimize this well.
 */

Vector& operator-= (Vector &lhs, Real rhs) {
    return lhs += -rhs;
}
Vector& operator/= (Vector &lhs, Real rhs) {
    return lhs *= 1/rhs;
}

Vector operator+ (Vector lhs, const Vector &rhs) {
    return lhs += rhs;
}
Vector operator+ (Vector lhs, Real rhs) {
    return lhs += rhs;
}
Vector operator+ (Real lhs, Vector rhs) {
    return rhs += lhs;
}
Vector operator-(Vector lhs, const Vector &rhs) {
    return lhs -= rhs;
}
Vector operator- (Vector lhs, Real rhs) {
    return lhs -= rhs;
}
Vector operator- (Real lhs, Vector rhs) {
    return (rhs -= lhs) *= -1;
}
Vector operator/ (Vector lhs, Real rhs) {
    return lhs /= rhs;
}
Vector operator* (Vector lhs, Real rhs) {
    return lhs *= rhs;
}
Vector operator* (Real lhs, Vector rhs) {
    return rhs *= lhs;
}
std::ostream& operator<<(std::ostream &strm, const Vector &b) {
    strm << "Vector(";
    for (int i = 0; i < b.get_dimensions() - 1; i++)
        strm << b[i] << ", ";
    strm << b[b.get_dimensions() - 1] << ")";
    return strm;
}
