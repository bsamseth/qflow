#include <vector>
#include <cassert>
#include <iostream>
#include "definitions.hpp"
#include "boson.hpp"

Boson::Boson(int dimensions) : _pos(dimensions, 0.0) { }

Boson::Boson(const Boson& from) : _pos(from._pos) { }

Boson::Boson(const std::vector<Real> &pos) : _pos(pos) { }

Real Boson::operator* (const Boson &other) const {
    assert(get_dimensions() == other.get_dimensions());
    Real prod = 0;
    for (int i = 0; i < get_dimensions(); i++) {
        prod += _pos[i] * other[i];
    }
    return prod;
}

Boson Boson::operator+ (const Boson &other) const {
    Boson res = *this;
    for (int i = 0; i < get_dimensions(); i++) {
        res[i] += other[i];
    }
    return res;
}

Boson Boson::operator- (const Boson &other) const {
    Boson res = *this;
    for (int i = 0; i < get_dimensions(); i++) {
        res[i] -= other[i];
    }
    return res;
}

Boson Boson::operator* (Real scalar) const {
    Boson res = *this;
    for (int i = 0; i < get_dimensions(); i++) {
        res[i] *= scalar;
    }
    return res;
}

Boson Boson::operator+ (Real scalar) const {
    Boson res = *this;
    for (int i = 0; i < get_dimensions(); i++) {
        res[i] += scalar;
    }
    return res;
}

Boson Boson::operator- (Real scalar) const {
    Boson res = *this;
    for (int i = 0; i < get_dimensions(); i++) {
        res[i] -= scalar;
    }
    return res;
}

Boson& Boson::operator+= (const Boson &other) {
    assert(get_dimensions() == other.get_dimensions());
    for (int i = 0; i < get_dimensions(); i++) {
        _pos[i] += other[i];
    }
    return *this;
}

Boson& Boson::operator-= (const Boson &other) {
    assert(get_dimensions() == other.get_dimensions());
    for (int i = 0; i < get_dimensions(); i++) {
        _pos[i] -= other[i];
    }
    return *this;
}

Boson& Boson::operator*= (Real scalar) {
    for (int i = 0; i < get_dimensions(); i++) {
        _pos[i] *= scalar;
    }
    return *this;
}

Boson& Boson::operator+= (Real scalar) {
    for (int i = 0; i < get_dimensions(); i++) {
        _pos[i] += scalar;
    }
    return *this;
}

Boson& Boson::operator-= (Real scalar) {
    for (int i = 0; i < get_dimensions(); i++) {
        _pos[i] -= scalar;
    }
    return *this;
}

std::ostream& operator<<(std::ostream &strm, const Boson &b) {
    strm << "Boson(";
    for (int i = 0; i < b.get_dimensions() - 1; i++)
        strm << b[i] << ", ";
    strm << b[b.get_dimensions() - 1] << ")";
    return strm;
}
