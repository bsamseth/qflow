#include "wavefunctionpooling.hpp"

SumPooling::SumPooling(Wavefunction& psi) : f(&psi)
{
    _parameters = f->get_parameters();
}

void SumPooling::set_parameters(const RowVector& a)
{
    f->set_parameters(a);
    _parameters = a;
}

Real SumPooling::operator()(const System& system)
{
    const int N   = system.rows();
    Real      sum = 0;
    System    sub(2, system.cols());
    for (int i = 0; i < N; ++i)
    {
        sub.row(0) = system.row(i);
        for (int j = 0; j < N; ++j)
        {
            if (i != j)
            {
                sub.row(1) = system.row(j);
                sum += (*f)(sub);
            }
        }
    }

    return sum;
}

RowVector SumPooling::gradient(const System& system)
{
    const int N    = system.rows();
    RowVector grad = RowVector::Zero(f->get_parameters().size());
    System    sub(2, system.cols());
    Real      divisor = 0;
    for (int i = 0; i < N; ++i)
    {
        sub.row(0) = system.row(i);
        for (int j = 0; j < N; ++j)
        {
            if (i != j)
            {
                sub.row(1) = system.row(j);
                Real eval  = (*f)(sub);
                grad += eval * f->gradient(sub);
                divisor += eval;
            }
        }
    }
    return grad / divisor;
}

Real SumPooling::drift_force(const System& system, int k, int dim_index)
{
    const int N     = system.rows();
    Real      drift = 0;
    System    sub(2, system.cols());

    sub.row(0) = system.row(k);
    for (int j = 0; j < N; ++j)
    {
        if (j != k)
        {
            sub.row(1) = system.row(j);
            drift += (*f)(sub) *f->drift_force(sub, 0, dim_index);
        }
    }

    sub.row(1) = system.row(k);
    for (int j = 0; j < N; ++j)
    {
        if (j != k)
        {
            sub.row(0) = system.row(j);
            drift += (*f)(sub) *f->drift_force(sub, 1, dim_index);
        }
    }
    return drift / (*this)(system);
}

Real SumPooling::laplacian(const System& system)
{
    const int N       = system.rows();
    Real      res     = 0;
    Real      divisor = 0;
    System    sub(2, system.cols());
    for (int i = 0; i < N; ++i)
    {
        sub.row(0) = system.row(i);
        for (int j = 0; j < N; ++j)
        {
            if (i != j)
            {
                sub.row(1) = system.row(j);
                Real eval  = (*f)(sub);
                res += eval * f->laplacian(sub);
                divisor += eval;
            }
        }
    }
    return res / divisor;
}
