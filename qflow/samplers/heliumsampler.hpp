#pragma once
#include "definitions.hpp"
#include "sampler.hpp"
#include "system.hpp"
#include "wavefunction.hpp"

#include <iostream>

class HeliumSampler : public Sampler
{
protected:
    /*
      From Sampler:

      const Real    _step;
      Wavefunction* _wavefunction;

      System _system_old;
      System _system_new;
      Real   _psi_old;
      Real   _psi_new;
      long   _accepted_steps   = 0;
      long   _total_steps      = 0;
      int    _particle_to_move = 0;

      void prepare_for_next_run();
    */

public:
    const Real L;
    const Real rho;

    HeliumSampler(const System& system, Wavefunction& psi, Real step, Real box_size);

    void initialize_system() override;

    void perturb_system() override;

    Real acceptance_probability() const override;
};
