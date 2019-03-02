
#pragma once

#include <Eigen/Dense>
#include "wavefunction.hpp"


class SumPooling : public Wavefunction {

private:
    Wavefunction* const f;

public:

    SumPooling(Wavefunction&);

    Real operator() (System&) override;
    RowVector gradient(System&) override;
    Real drift_force(const System &system, int k, int dim_index) override;
    Real laplacian(System&) override;

    void set_parameters(const RowVector&) override;
};
