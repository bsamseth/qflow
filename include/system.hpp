#pragma once

#include <vector>
#include <iostream>
#include "boson.hpp"


class System {

    std::vector<Boson> _bosons;

    public:

        System(int number_of_bosons, int dimensions);

        Boson& operator[] (int index);

        std::vector<Boson>& get_bosons();
};

inline Boson& System::operator[] (int index) {
    return _bosons[index];
}

inline std::vector<Boson>& System::get_bosons() {
    return _bosons;
}

inline std::ostream& operator<<(std::ostream &strm, System &s) {
    strm << "System(";
    for (int i = 0; i < s.get_bosons().size() - 1; i++)
        strm << s[i] << ", ";
    strm << s[s.get_bosons().size() - 1] << ")";
    return strm;
}

