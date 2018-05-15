#include <vector>
#include <cassert>
#include "vector.hpp"
#include "system.hpp"

System::System(int number_of_particles, int dimensions) :
    _particles(number_of_particles, Vector{dimensions}),
    _distances(number_of_particles, std::vector<Real>(number_of_particles, 0)),
    _dirty(number_of_particles, std::vector<bool>(number_of_particles, true))
{ }

Real System::distance(int i, int j) {
    assert(0 <= i and i < get_n_particles());
    assert(0 <= j and j < get_n_particles());
    if (_dirty[i][j]) {
        _distances[i][j] = _distances[j][i] = (_particles[i] - _particles[j]).norm();
        _dirty[i][j] = _dirty[j][i] = false;
    }
    return _distances[i][j];
}

Vector& System::operator() (int index) {
    assert(0 <= index and index < get_n_particles());
    auto n = get_n_particles();
    for (int i = 0; i < n; ++i)
        _dirty[index][i] = _dirty[i][index] = true;
    return _particles[index];
}
