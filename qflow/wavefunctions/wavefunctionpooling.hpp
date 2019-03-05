#pragma once

#include "wavefunction.hpp"

#include <Eigen/Dense>

class SumPooling : public Wavefunction
{
private:
    Wavefunction* const f;

public:
    explicit SumPooling(Wavefunction&);

    Real      operator()(const System&) override;
    RowVector gradient(const System&) override;
    Real      drift_force(const System& system, int k, int dim_index) override;
    Real      laplacian(const System&) override;

    void set_parameters(const RowVector&) override;
};
