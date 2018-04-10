#include <vector>
#include <cassert>
#include "boson.hpp"
#include "system.hpp"

System::System(int number_of_bosons, int dimensions) :
    _bosons(number_of_bosons, Boson{dimensions}),
    _distances(number_of_bosons, std::vector<Real>(number_of_bosons, 0)),
    _dirty(number_of_bosons, true)
{ }

Real System::distance(int i, int j) {
    assert(0 <= i and i < get_n_bosons());
    assert(0 <= j and j < get_n_bosons());
    if (_dirty[i]) {
        auto n = get_n_bosons();
        const auto &bi = _bosons[i];
        for (int k = 0; k < n; ++k) {
            _distances[i][k] = _distances[k][i] = std::sqrt(square(bi - _bosons[k]));
        }
        _dirty[i] = false;
    }
    if (_dirty[j]) {
        auto n = get_n_bosons();
        const auto &bj = _bosons[j];
        for (int k = 0; k < n; ++k) {
            _distances[k][j] = _distances[j][k] = std::sqrt(square(bj - _bosons[k]));
        }
        _dirty[j] = false;
    }
    return _distances[i][j];
}
