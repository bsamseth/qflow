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
        std::vector<Real>&  get_position();
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
        const Real& operator[] (int dimension) const;
        /**
         * Dot product of two bosons.
         * @param other Rhs. of the multiplication.
         * @return The dot product of this and other.
         */
        Real operator* (const Boson& other) const;
        /**
         * Elementwise multiplication by scalar.
         * @param scalar Scalar value to multiply with.
         * @return A boson with coordinates scaled by scalar.
         */
        Boson operator* (Real scaler) const;
        /**
         * Vector addition of two bosons.
         * @param other Rhs. of the addition.
         * @return Vector sum of this and other.
         */
        Boson operator+ (const Boson &other) const;
        /**
         * Elemtwise addition by scalar.
         * @param scalar Scalar to add to boson.
         * @return A boson with coordinate values added by scalar.
         */
        Boson operator+ (Real scalar) const;
        /**
         * Vector subtraction of two bosons.
         * @param other Rhs. of the subtraction.
         * @return Vector difference of this and other.
         */
        Boson operator- (const Boson &other) const;
        /**
         * Elemtwise subtraction by scalar.
         * @param scalar Scalar to subtract from boson.
         * @return A boson with coordinate values subtracted by scalar.
         */
        Boson operator- (Real) const;
        /**
         * Equivalency test.
         * @param other Boson to compare with.
         * @return Result of `==` applied to both bosons internal representation.
         */
        bool operator== (const Boson &other) const;
        /**
         * Negated equivalency test.
         * @param other Boson to compare with.
         * @return Result of `!=` applied to both bosons internal representation.
         */
        bool operator!= (const Boson &other) const;
        /**
         * Inplace vector addition of two bosons.
         * @param other Rhs. of the addition.
         * @return Reference to lhs., which has been updated with the vector sum.
         */
        Boson& operator+= (const Boson &other);
        /**
         * Inplace elementwise addition of boson with scalar
         * @param scalar Rhs. of the addition.
         * @return Reference to lhs., which has been updated with the elementwise sum.
         */
        Boson& operator+= (Real scalar);
        /**
         * Inplace elementwise multiplication boson with scalar.
         * @param scalar Rhs. of the multiplication.
         * @return Reference to lhs., which has been updated with the elementwise multiplication.
         */
        Boson& operator*= (Real scalar);
        /**
         * Inplace vector subtraction of two bosons.
         * @param other Rhs. of the subtraction.
         * @return Reference to lhs., which has been updated with the vector difference.
         */
        Boson& operator-= (const Boson &other);
        /**
         * Inplace elementwise subtraction of boson with scalar
         * @param scalar Rhs. of the addition.
         * @return Reference to lhs., which has been updated with the elementwise sum.
         */
        Boson& operator-= (Real scalar);

        /**
         * Stream text representation of the Boson to a stream.
         */
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

