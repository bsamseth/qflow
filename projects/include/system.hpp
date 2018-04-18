#pragma once

#include <vector>
#include <cassert>
#include <iostream>
#include "boson.hpp"


/**
 * Class representing a system of Bosons.
 */
class System {

    std::vector<Boson> _bosons;
    std::vector<std::vector<Real>> _distances;
    std::vector<std::vector<bool>> _dirty;

    public:

        /**
         * Initialize a system of N D-dimensional bosons, placed at origin.
         * @param number_of_bosons N
         * @param dimensions D
         */
        System(int number_of_bosons, int dimensions);
        /**
         * @param index Index of Boson.
         * @return Boson at given index.
         */
        const Boson& operator[] (int index) const;

        Boson& operator() (int index);

        Real degree(int k) const;

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
         * @return Internal representation of all Bosons.
         */
        const std::vector<Boson>& get_bosons() const;

        const std::vector<std::vector<bool> > get_dirty() const;

        /**
         * @return Number of dimensions of the System.
         */
        int get_dimensions() const;

        /**
         * @return Number of Bosons in the System.
         */
        int get_n_bosons() const;
};

inline const Boson& System::operator[] (int index) const {
    assert(0 <= index and index < get_n_bosons());
    return _bosons[index];
}
inline Real System::degree(int k) const {
    return _bosons[k / get_dimensions()][k % get_dimensions()];
}
inline bool System::operator== (const System &other) const {
    return _bosons == other.get_bosons();
}
inline bool System::operator!= (const System &other) const {
    return _bosons != other.get_bosons();
}
inline const std::vector<Boson>& System::get_bosons() const {
    return _bosons;
}
inline const std::vector<std::vector<bool>> System::get_dirty() const {
    return _dirty;
}
inline int System::get_dimensions() const {
    assert(get_n_bosons() > 0);
    return _bosons[0].get_dimensions();
}
inline int System::get_n_bosons() const {
    return _bosons.size();
}
inline std::ostream& operator<<(std::ostream &strm, const System &s) {
    strm << "System(";
    for (int i = 0; i < s.get_n_bosons() - 1; i++)
        strm << s[i] << ", ";
    strm << s[s.get_n_bosons() - 1] << ")";
    return strm;
}

