#include <vector>
#include "system.hpp"

System::System(int number_of_bosons, int dimensions) {
    for (int i = 0; i < number_of_bosons; ++i) {
        _bosons.push_back( { dimensions } );
    }
}

