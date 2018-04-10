#pragma once

#include <vector>
#include <iostream>
#include "definitions.hpp"

/**
 * N-dimensional vector representation of a boson.
 */
class Boson {
    private:
        std::vector<Real> _pos;

    public:

        /**
         * Initialize a Boson with a given number of dimensions, initialized to origo.
         * @param dimensions Number of dimensions to use.
         */
        Boson(int dimensions);
        /**
         * Initialize a Boson from a std::vector.
         * @param vec Vector of initialization values.
         */
        Boson(const std::vector<Real> &vec);
        /**
         * @return Internal position vector.
         */
        const std::vector<Real>& get_position() const;
        /**
         * @return Number of dimensions of the boson.
         */
        int get_dimensions() const;
        /**
         * @param dimensions The coordinate to get.
         * @return Reference to value for the given coordinate.
         */
        Real& operator[] (int dimension);
        /**
         * @param dimensions The coordinate to get.
         * @return Reference to value for the given coordinate.
         */
        Real operator[] (int dimension) const;

        /**
         * Stream text representation of the Boson to a stream.
         */
        friend std::ostream& operator<<(std::ostream&, const Boson&);
};

inline Real& Boson::operator[] (int dimension) {
    return _pos[dimension];
}
inline Real Boson::operator[] (int dimension) const {
    return _pos[dimension];
}
inline const std::vector<Real>& Boson::get_position() const {
    return _pos;
}
inline int Boson::get_dimensions() const {
    return _pos.size();
}

/*
 * @param lhs Lhs. of the equality check.
 * @param rhs Rhs. of the equality check.
 * @return Result of `==` on the underlying container.
 */
inline bool operator== (const Boson &lhs, const Boson &rhs) {
    return lhs.get_position() == rhs.get_position();
}
/*
 * @return Result of `!(lhs == rhs)`.
 */
inline bool operator!= (const Boson &lhs, const Boson &rhs) {
    return !(lhs == rhs);
}


/*
 * All the following operator functions are defined as one would
 * expect when interpreting Bosons as mathematical vectors.
 *
 *   Boson (+-) Boson  => Elementwise sum/difference.
 *   Boson * Boson     => Inner product
 *   Boson (+*-/) Real => Elementwise add/mult/sub/div by scalar.
 *
 * For the sake of breviety, no further documentation is therefore
 * provided for each operator declaration.
 */

Real operator* (const Boson &lhs, const Boson& rhs);
Boson operator* (Boson lhs, Real rhs);
Boson operator+ (Boson lhs, const Boson &rhs);
Boson operator+ (Boson lhs, Real rhs);
Boson operator+ (Real lhs, Boson rhs);
Boson operator- (Boson lhs, const Boson &rhs);
Boson operator- (Boson lhs, Real rhs);
Boson operator- (Real lhs, Boson rhs);
Boson operator/ (Boson lhs, Real rhs);
Boson operator/ (Real lhs, Boson rhs);

Boson& operator+= (Boson &lhs, const Boson &rhs);
Boson& operator+= (Boson &lhs, Real rhs);
Boson& operator*= (Boson &lhs, Real rhs);
Boson& operator-= (Boson &lhs, const Boson &rhs);
Boson& operator-= (Boson &lhs, Real rhs);
Boson& operator/= (Boson &lhs, Real rhs);



