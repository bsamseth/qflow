#pragma once

#include "system.hpp"
#include "calculator.hpp"

/**
 * Template class used to combine calculators into one.
 *
 * Given two calculators, this class represents the effective result
 * of implementing all calculations done in First _and_ Second in a new
 * combined calculator. Combinations can be made with other combinations,
 * allowing arbitrarily nested calculators.
 */
template<class First, class Second>
class CombinedCalculator : public Calculator {

    protected:
        First &_first;
        Second &_second;

    public:

        /**
         * Initialize a new calculator as the combination of first and second.
         *
         * The two samplers must have the same wavefunction, Hamiltonian, sampler and logfile name.
         * @param first Instance of a calculator.
         * @param second Instance of another calculator.
         */
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

        /**
         * Process the state using both internal calculators.
         * @param state System sample to process.
         */
        virtual void process_state(const System &state) {
            _first.process_state(state);
            _second.process_state(state);
        }

        /**
         * Finalize both internal calculators.
         */
        virtual void finalize_calculation() {
            _first.finalize_calculation();
            _second.finalize_calculation();
        }
};
