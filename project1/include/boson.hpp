#pragma once

#include <vector>
#include <iostream>
#include "definitions.hpp"

class Boson {
    private:
        std::vector<Real> _pos;

    public:

        Boson(int dimensions);
        Boson(const Boson&);
        Boson(const std::vector<Real>&);
        std::vector<Real>&  get_position();
        int get_dimensions() const;
        Real& operator[] (int dimension);
        const Real& operator[] (int dimension) const;
        Real operator* (const Boson&) const;
        Boson operator* (Real) const;
        Boson operator+ (const Boson&) const;
        Boson operator+ (Real) const;
        Boson operator- (const Boson&) const;
        Boson operator- (Real) const;
        bool operator== (const Boson&) const;
        bool operator!= (const Boson&) const;
        Boson& operator+= (const Boson&);
        Boson& operator+= (Real);
        Boson& operator*= (Real);
        Boson& operator-= (const Boson&);
        Boson& operator-= (Real);
        friend std::ostream& operator<<(std::ostream&, const Boson&);
};

inline Real& Boson::operator[] (int dimension) {
    return _pos[dimension];
}
inline const Real& Boson::operator[] (int dimension) const {
    return _pos[dimension];
}
inline bool Boson::operator== (const Boson& other) const {
    return _pos == other._pos;
}
inline bool Boson::operator!= (const Boson& other) const {
    return _pos != other._pos;
}
inline std::vector<Real>& Boson::get_position() {
    return _pos;
}
inline int Boson::get_dimensions() const {
    return _pos.size();
}

