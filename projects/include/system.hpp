#pragma once

#include <vector>
#include <cassert>
#include <iostream>
#include "vector.hpp"


/**
 * Class representing a system of particles.
 */
class System {

    std::vector<Vector> _particles;
    std::vector<std::vector<Real>> _distances;
    std::vector<std::vector<bool>> _dirty;

    public:

        /**
         * Initialize a system of N D-dimensional particles, placed at origin.
         * @param number_of_particles N
         * @param dimensions D
         */
        System(int number_of_particles, int dimensions);
        /**
         * @param index Index of particle.
         * @return particle at given index.
         */
        const Vector& operator[] (int index) const;

        Vector& operator() (int index);

        Real degree(int k) const;
        Real& degree(int k);

        Real distance(int i, int j);

        /**
         * Equivalency operation.
         * @param other System to compare with.
         * @return Result of `==` on the internal representation of both systems.
         */
        bool operator== (const System &other) const;
        /**
         * Negative equivalency operation.
         * @param other System to compare with.
         * @return Result of `!=` on the internal representation of both systems.
         */
        bool operator!= (const System &other) const;

        /**
         * @return Internal representation of all particles.
         */
        const std::vector<Vector>& get_particles() const;

        const std::vector<std::vector<bool> > get_dirty() const;

        /**
         * @return Number of dimensions of the System.
         */
        int get_dimensions() const;

        /**
         * @return Number of particles in the System.
         */
        int get_n_particles() const;
};

inline const Vector& System::operator[] (int index) const {
    assert(0 <= index and index < get_n_particles());
    return _particles[index];
}
inline Real System::degree(int k) const {
    return _particles[k / get_dimensions()][k % get_dimensions()];
}
inline Real& System::degree(int k) {
    return (*this)(k / get_dimensions())[k % get_dimensions()];
}
inline bool System::operator== (const System &other) const {
    return _particles == other.get_particles();
}
inline bool System::operator!= (const System &other) const {
    return _particles != other.get_particles();
}
inline const std::vector<Vector>& System::get_particles() const {
    return _particles;
}
inline const std::vector<std::vector<bool>> System::get_dirty() const {
    return _dirty;
}
inline int System::get_dimensions() const {
    assert(get_n_particles() > 0);
    return _particles[0].get_dimensions();
}
inline int System::get_n_particles() const {
    return _particles.size();
}
inline std::ostream& operator<<(std::ostream &strm, const System &s) {
    strm << "System(";
    for (int i = 0; i < s.get_n_particles() - 1; i++)
        strm << s[i] << ", ";
    strm << s[s.get_n_particles() - 1] << ")";
    return strm;
}

