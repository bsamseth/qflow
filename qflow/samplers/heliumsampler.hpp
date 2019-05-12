#pragma once
#include "definitions.hpp"
#include "sampler.hpp"
#include "system.hpp"
#include "wavefunction.hpp"

#include <iostream>

class HeliumSampler : public Sampler
{
protected:
public:
    const Real L;

    HeliumSampler(const System& system, Wavefunction& psi, Real step, Real box_size);

    void initialize_system() override;

    void perturb_system() override;

    Real acceptance_probability() const override;
};
