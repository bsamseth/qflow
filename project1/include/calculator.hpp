#pragma once

#include <string>

#include "system.hpp"
#include "wavefunction.hpp"
#include "hamiltonian.hpp"
#include "sampler.hpp"

/**
 * Abstract base class for a generic property calculator.
 *
 * The concept of a Calculator is to process a stream of sampled
 * System instances produced by a Sampler, and calculate some desired
 * quantity as a function of this.
 *
 * Using instantiations of CombinedCalculator, we can combine an arbitrary
 * set of calculators as one. The benefit of this is that each sampled System
 * instance can be processes by each calculator, without the need to
 * copy/store/recalculate them for use in other calculations.
 *
 * This class provided the interface subclasses are expected to follow.
 */
class Calculator {

    public:

        const Wavefunction &_wavefunction;
        const Hamiltonian &_hamiltonian;
        Sampler &_sampler;
        const std::string _logfile_name;


        /**
         * Instantiate a calculator.
         * @param wavefunction Wavefunction to calculate with.
         * @param hamiltonian Hamiltonian to calculate with.
         * @param sampler Source of sampled System instances.
         * @param logfile Name of file to log output to.
         */
        Calculator(const Wavefunction &wavefunction,
                   const Hamiltonian &hamiltonian,
                   Sampler &sampler,
                   std::string logfile);

        /**
         * Process a given number of sampled System instances, logging the result as specified by class.
         * @param iterations Number of samples to process.
         */
        virtual void calculate(long iterations);
        /**
         * Perform calculations on the System sample.
         * @param system Sample to process.
         */
        virtual void process_state(const System &system) = 0;
        /**
         * Perform any closing logic needed (e.g. close output files).
         */
        virtual void finalize_calculation() = 0;

};
