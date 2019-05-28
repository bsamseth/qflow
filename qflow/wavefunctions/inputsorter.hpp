#pragma once

#include "wavefunction.hpp"

#include <Eigen/Dense>

class InputSorter : public Wavefunction
{
private:
    Wavefunction* const f;

public:
    InputSorter(Wavefunction& psi) : f(&psi) {}

    Real      operator()(const System&) override;
    RowVector gradient(const System&) override;
    Real      drift_force(const System& system, int k, int dim_index) override;
    Real      laplacian(const System&) override;

    void             set_parameters(const RowVector&) override;
    const RowVector& get_parameters() const override;
};

const RowVector& InputSorter::get_parameters() const
{
    return f->get_parameters();
}

void InputSorter::set_parameters(const RowVector& a)
{
    f->set_parameters(a);
}
