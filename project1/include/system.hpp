#pragma once

#include <vector>
#include <iostream>
#include "boson.hpp"


class System {

    std::vector<Boson> _bosons;

    public:

        System(int number_of_bosons, int dimensions);
        System(const System&);

        Boson& operator[] (int index);
        const Boson& operator[] (int index) const;
        bool operator== (const System&) const;
        bool operator!= (const System&) const;

        std::vector<Boson>& get_bosons();
        const std::vector<Boson>& get_bosons() const;

        int get_dimensions() const;

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

