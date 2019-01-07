#include "definitions.hpp"
#include "wavefunction.hpp"

Wavefunction::Wavefunction(std::initializer_list<Real> parameters) :
        _parameters(vector_from_sequence(parameters)) {}

