#include "wavefunctionpooling.hpp"

SumPooling::SumPooling(Wavefunction& psi)
    : f(&psi)
{
    _parameters = f->get_parameters();
}

void SumPooling::set_parameters(const RowVector& a) {
    f->set_parameters(a);
    _parameters = a;
}


Real SumPooling::operator() (System& system) {
    const int N = system.rows();
    Real sum = 0;
    System sub(2, system.cols());
    for (int k = 0; k < N; ++k) {
        sub.row(0) = system.row(k);
        for (int j = 0; j < N; ++j) {
            sub.row(1) = system.row(j);
            sum += (*f)(sub);
        }
    }

    return sum;
}

RowVector SumPooling::gradient(System& system) {
    const int N = system.rows();
    RowVector grad = RowVector::Zero(f->get_parameters().size());
    System sub(2, system.cols());
    for (int k = 0; k < N; ++k) {
        sub.row(0) = system.row(k);
        for (int j = 0; j < N; ++j) {
            sub.row(1) = system.row(j);
            grad += (*f)(sub) * f->gradient(sub);
        }
    }
    return grad / (*this)(system);
}

Real SumPooling::drift_force(const System &system, int k, int dim_index) {
    const int N = system.rows();
    Real drift = 0;
    System sub(2, system.cols());
    sub.row(0) = system.row(k);
    for (int j = 0; j < N; ++j) {
        sub.row(1) = system.row(j);
        drift += (*f)(sub) * f->drift_force(sub, k, dim_index);
    }
    sub.row(1) = system.row(k);
    for (int j = 0; j < N; ++j) {
        sub.row(0) = system.row(j);
        drift += (*f)(sub) * f->drift_force(sub, k, dim_index);
    }
    return drift / (*this)(system);
}

Real SumPooling::laplacian(System &system) {
    const int N = system.rows();
    Real res = 0;
    System sub(2, system.cols());
    for (int k = 0; k < N; ++k) {
        sub.row(0) = system.row(k);
        for (int j = 0; j < N; ++j) {
            sub.row(1) = system.row(j);
            res += (*f)(sub) * f->laplacian(sub);
        }
    }
    return res / (*this)(system);
}
