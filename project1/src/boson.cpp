#include <vector>
#include <cassert>
#include <iostream>
#include "definitions.hpp"
#include "boson.hpp"

Boson::Boson(int dimensions) : _pos(dimensions, 0.0) {
    // Dimension expected to be in [1, 3]. Nothing about the
    // implementation should break for Dim >= 4, but if this is
    // used, it is more likely a programmer typo/bug.
    assert(dimensions > 0 and dimensions < 4);
}

/*
 * Allowed for simplicity in creating test instances.
 */
Boson::Boson(const std::vector<Real> &pos) : _pos(pos) { }

Real operator* (const Boson &lhs, const Boson &rhs) {
    assert(lhs.get_dimensions() == rhs.get_dimensions());
    Real prod = 0;
    for (int i = 0; i < lhs.get_dimensions(); i++) {
        prod += lhs[i] * rhs[i];
    }
    return prod;
}
Boson& operator+= (Boson &lhs, const Boson &rhs) {
    assert(lhs.get_dimensions() == rhs.get_dimensions());
    for (int i = 0; i < lhs.get_dimensions(); i++) {
        lhs[i] += rhs[i];
    }
    return lhs;
}
Boson& operator-= (Boson &lhs, const Boson &rhs) {
    assert(lhs.get_dimensions() == rhs.get_dimensions());
    for (int i = 0; i < lhs.get_dimensions(); i++) {
        lhs[i] -= rhs[i];
    }
    return lhs;
}
Boson& operator+= (Boson &lhs, Real rhs) {
    for (int i = 0; i < lhs.get_dimensions(); i++) {
        lhs[i] += rhs;
    }
    return lhs;
}
Boson& operator*= (Boson &lhs, Real rhs) {
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

Boson& operator-= (Boson &lhs, Real rhs) {
    return lhs += -rhs;
}
Boson& operator/= (Boson &lhs, Real rhs) {
    return lhs *= 1/rhs;
}

Boson operator+ (Boson lhs, const Boson &rhs) {
    return lhs += rhs;
}
Boson operator+ (Boson lhs, Real rhs) {
    return lhs += rhs;
}
Boson operator+ (Real lhs, Boson rhs) {
    return rhs += lhs;
}
Boson operator-(Boson lhs, const Boson &rhs) {
    return lhs -= rhs;
}
Boson operator- (Boson lhs, Real rhs) {
    return lhs -= rhs;
}
Boson operator- (Real lhs, Boson rhs) {
    return (rhs -= lhs) *= -1;
}
Boson operator/ (Boson lhs, Real rhs) {
    return lhs /= rhs;
}
Boson operator* (Boson lhs, Real rhs) {
    return lhs *= rhs;
}
Boson operator* (Real lhs, Boson rhs) {
    return rhs *= lhs;
}
std::ostream& operator<<(std::ostream &strm, const Boson &b) {
    strm << "Boson(";
    for (int i = 0; i < b.get_dimensions() - 1; i++)
        strm << b[i] << ", ";
    strm << b[b.get_dimensions() - 1] << ")";
    return strm;
}
