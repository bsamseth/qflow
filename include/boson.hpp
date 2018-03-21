#pragma once

#include <vector>
#include <iostream>
#include "definitions.hpp"

class Boson {
    std::vector<Real> _pos;

    public:

    Boson(int dimensions);

    Real& operator[] (int dimension);

    std::vector<Real>&  get_position();

    friend std::ostream& operator<<(std::ostream&, const Boson&);
};

inline Real& Boson::operator[] (int dimension) {
    return _pos[dimension];
}

inline std::vector<Real>& Boson::get_position() {
    return _pos;
}

inline std::ostream& operator<<(std::ostream &strm, Boson &b) {
    strm << "Boson(";
    for (int i = 0; i < b.get_position().size() - 1; i++)
        strm << b[i] << ", ";
    strm << b[b.get_position().size() - 1] << ")";
    return strm;
}

