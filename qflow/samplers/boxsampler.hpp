#pragma once

#include "distance.hpp"
#include "gibbssampler.hpp"
#include "importancesampler.hpp"
#include "metropolissampler.hpp"

#include <iostream>
#include <limits>

template <class Base>
class BoxSampler : public Base
{
private:
    Real L_;

public:
    BoxSampler(const System& init, Wavefunction& psi, Real box_size, Real step)
        : Base(init, psi, step), L_(box_size)
    {
        initialize_system();
    }

    System& next_configuration() override
    {
        System& s = Base::next_configuration();
        s -= (Eigen::floor(s.array() / L_) * L_).matrix();
        return s;
    }

    void initialize_from_system(const System& system)
    {
        Base::_system_old = system;
        Base::_system_new = Base::_system_old;
        Base::_psi_old    = (*Base::_wavefunction)(Base::_system_old);
    }

    void initialize_system() override
    {
        Distance::stop_tracking();
        Distance::set_simulation_box_size(L_);
        std::uniform_real_distribution<Real> dist(0, L_);

        for (int i = 0; i < Base::_system_old.rows(); ++i)
        {
            RowVector best_point = RowVector::Zero(Base::_system_old.cols());
            Real      best_dist  = std::numeric_limits<Real>::min();
            for (int run = 0; run < 10000; run++)
            {
                // Pick a new point at random
                for (int d = 0; d < Base::_system_old.cols(); ++d)
                {
                    Base::_system_old(i, d) = dist(rand_gen);
                }

                if (i == 0)
                {
                    best_point = Base::_system_old;
                    break;
                }

                /* std::cout << "\tproposed point: " << Base::_system_old.row(i); */
                Real min_dist = std::numeric_limits<Real>::max();
                // Compute minimum distance to all other set points:
                for (int j = 0; j < i; ++j)
                {
                    Real r_ij = Distance::probe(Base::_system_old, i, j);
                    if (r_ij < min_dist)
                        min_dist = r_ij;
                }

                /* std::cout << " with min dist = " << min_dist <<std::endl; */
                if (min_dist > best_dist)
                {
                    best_dist  = min_dist;
                    best_point = Base::_system_old.row(i);
                }
            }
            Base::_system_old.row(i) = best_point;
            /* std::cout << "Placed point at " << Base::_system_old.row(i) << " with min
             * dist " << best_dist << std::endl; */
        }
        Base::_system_new = Base::_system_old;
        Base::_psi_old    = (*Base::_wavefunction)(Base::_system_old);
    }
};

using BoxMetropolisSampler = BoxSampler<MetropolisSampler>;
using BoxImportanceSampler = BoxSampler<ImportanceSampler>;
