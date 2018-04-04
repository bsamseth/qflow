#pragma once

#include <vector>
#include <iostream>
#include "boson.hpp"


/**
 * Class representing a system of Bosons.
 */
class System {

    std::vector<Boson> _bosons;

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
        Boson& operator[] (int index);
        /**
         * @param index Index of Boson.
         * @return Boson at given index.
         */
        const Boson& operator[] (int index) const;
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
        std::vector<Boson>& get_bosons();
        /**
         * @return Internal representation of all Bosons.
         */
        const std::vector<Boson>& get_bosons() const;

        /**
         * @return Number of dimensions of the System.
         */
        int get_dimensions() const;

        /**
         * @return Number of Bosons in the System.
         */
        int get_n_bosons() const;
};

inline Boson& System::operator[] (int index) {
    return _bosons[index];
}
inline const Boson& System::operator[] (int index) const {
    return _bosons[index];
}
inline bool System::operator== (const System &other) const {
    return _bosons == other.get_bosons();
}
inline bool System::operator!= (const System &other) const {
    return _bosons != other.get_bosons();
}
inline std::vector<Boson>& System::get_bosons() {
    return _bosons;
}
inline const std::vector<Boson>& System::get_bosons() const {
    return _bosons;
}
inline int System::get_dimensions() const {
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

