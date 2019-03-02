#pragma once
#include "sampler.hpp"

#include <iostream>
#include <memory>

/**
 * Parallel sampler class for Monte Carlo sampling using MPI.
 *
 * Note: This class does not utilize MPI it self, but is simply a convenience
 * container for a set of samplers.
 */
template <typename BackendSampler>
class PSampler
{
protected:
    std::vector<BackendSampler> samplers;

public:
    PSampler(const BackendSampler& init, std::size_t N)
    {
        // Make N copies.
        for (std::size_t i = 0; i < N; ++i)
            samplers.push_back(init);

        // Initialize all copies randomly.
        for (std::size_t i = 0; i < N; ++i)
            samplers[i].initialize_system();
    }

    BackendSampler& operator[](std::size_t i);
};

template <typename BackendSampler>
inline BackendSampler& operator[](std::size_t i)
{
    assert(i < samplers.size());
    return samplers[i];
}
