#pragma once

#include "system.hpp"
#include "calculator.hpp"

template<class First, class Second>
class CombinedCalculator : public Calculator {

    protected:
        First &_first;
        Second &_second;

    public:

        CombinedCalculator(First &first, Second &second) : Calculator(first._wavefunction,
                                                                      first._hamiltonian,
                                                                      first._sampler,
                                                                      first._logfile_name),
                                                            _first(first), _second(second)
        {
            // Design flaw makes it neccessary to initialize this using first/second properties.
            // They should not differ though, so we can double check this for consistency.
            assert(&first._wavefunction == &second._wavefunction);
            assert(&first._hamiltonian == &second._hamiltonian);
            assert(&first._sampler == &second._sampler);
            assert(first._logfile_name == second._logfile_name);
        }

        virtual void process_state(const System &state) {
            _first.process_state(state);
            _second.process_state(state);
        }

        virtual void finalize_calculation() {
            _first.finalize_calculation();
            _second.finalize_calculation();
        }
};
