#include <vector>
#include <cassert>
#include "boson.hpp"
#include "system.hpp"

System::System(int number_of_bosons, int dimensions) :
    _bosons(number_of_bosons, Boson{dimensions}),
    _distances(number_of_bosons, std::vector<Real>(number_of_bosons, 0)),
    _dirty(number_of_bosons, std::vector<bool>(number_of_bosons, true))
{ }

Real System::distance(int i, int j) {
    assert(0 <= i and i < get_n_bosons());
    assert(0 <= j and j < get_n_bosons());
    if (_dirty[i][j]) {
        _distances[i][j] = _distances[j][i] = std::sqrt(square(_bosons[i] - _bosons[j]));
        _dirty[i][j] = _dirty[j][i] = false;
    }
    return _distances[i][j];
}

Boson& System::operator() (int index) {
    assert(0 <= index and index < get_n_bosons());
    auto n = get_n_bosons();
    for (int i = 0; i < n; ++i)
        _dirty[index][i] = _dirty[i][index] = true;
    return _bosons[index];
}
