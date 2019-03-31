#pragma once

#include "gibbssampler.hpp"
#include "importancesampler.hpp"
#include "metropolissampler.hpp"

template <class Base>
class BoxSampler : public Base
{
private:
    Real L_;

public:
    BoxSampler(const System& init, Wavefunction& psi, Real box_size, Real step)
        : Base(init, psi, step), L_(box_size)
    {
    }

    System& next_configuration() override
    {
        System& s = Base::next_configuration();
        s -= (Eigen::floor(s.array() / L_) * L_).matrix();
        return s;
    }
};

using BoxMetropolisSampler = BoxSampler<MetropolisSampler>;
using BoxImportanceSampler = BoxSampler<ImportanceSampler>;
