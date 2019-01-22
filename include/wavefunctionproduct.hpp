#pragma once

#include <Eigen/Dense>
#include "wavefunction.hpp"


class WavefunctionProduct : public Wavefunction {

private:
    Wavefunction* const f;
    Wavefunction* const g;

public:

    WavefunctionProduct(Wavefunction&, Wavefunction&);

    Real operator() (System&) override;
    RowVector gradient(System&) override;
    Real drift_force(const System &system, int k, int dim_index) override;
    Real laplacian(System&) override;

    void set_parameters(const RowVector&) override;
    void set_parameters(const RowVector&, const RowVector&);

};

inline Real WavefunctionProduct::operator() (System& system) {
    return (*f)(system) * (*g)(system);
}

inline Real WavefunctionProduct::drift_force(const System& system, int k, int dim_index) {
    return f->drift_force(system, k, dim_index) + g->drift_force(system, k, dim_index);
}

inline Real WavefunctionProduct::laplacian(System& system) {
    return f->laplacian(system) + g->laplacian(system) + 0.5 * f->drift_force(system).dot(g->drift_force(system));
}
