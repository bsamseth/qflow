#include "wavefunctionpooling.hpp"

#include "omp.h"

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
    const int N  = system.rows();
    Real      s1 = 0;
    Real      s2 = 0;
    System    sub1(2, system.cols());
    System    sub2(2, system.cols());
    for (int i = 0; i < N; ++i)
    {
        sub1.row(0) = system.row(i);
        sub2.row(0) = system.row(i);
#pragma omp parallel sections
        {
#pragma omp section
            {
                for (int j = 0; j < i; ++j)
                {
                    sub1.row(1) = system.row(j);
                    s1 += (*f)(sub1);
                }
            }
#pragma omp section
            {
                for (int j = i + 1; j < N; ++j)
                {
                    sub2.row(1) = system.row(j);
                    s2 += (*f)(sub2);
                }
            }
        }
    }
    return s1 + s2;
}

RowVector SumPooling::gradient(const System& system)
{
    const int N  = system.rows();
    RowVector g1 = RowVector::Zero(f->get_parameters().size());
    RowVector g2 = RowVector::Zero(f->get_parameters().size());
    System    sub1(2, system.cols());
    System    sub2(2, system.cols());
    Real      d1 = 0;
    Real      d2 = 0;
    for (int i = 0; i < N; ++i)
    {
        sub1.row(0) = system.row(i);
        sub2.row(0) = system.row(i);
#pragma omp parallel sections
        {
#pragma omp section
            {
                for (int j = 0; j < i; ++j)
                {
                    sub1.row(1) = system.row(j);
                    Real eval   = (*f)(sub1);
                    g1 += eval * f->gradient(sub1);
                    d1 += eval;
                }
            }
#pragma omp section
            {
                for (int j = i + 1; j < N; ++j)
                {
                    sub2.row(1) = system.row(j);
                    Real eval   = (*f)(sub2);
                    g2 += eval * f->gradient(sub2);
                    d2 += eval;
                }
            }
        }
    }
    return (g1 + g2) / (d1 + d2);
}

Real SumPooling::drift_force(const System& system, int k, int dim_index)
{
    const int N  = system.rows();
    Real      d1 = 0;
    Real      d2 = 0;
    System    sub1(2, system.cols());
    System    sub2(2, system.cols());
#pragma omp parallel sections
    {
#pragma omp section
        {
            sub1.row(0) = system.row(k);
            for (int j = 0; j < N; ++j)
            {
                if (j != k)
                {
                    sub1.row(1) = system.row(j);
                    d1 += (*f)(sub1) *f->drift_force(sub1, 0, dim_index);
                }
            }
        }
#pragma omp section
        {
            sub2.row(1) = system.row(k);
            for (int j = 0; j < N; ++j)
            {
                if (j != k)
                {
                    sub2.row(0) = system.row(j);
                    d2 += (*f)(sub2) *f->drift_force(sub2, 1, dim_index);
                }
            }
        }
    }
    return (d1 + d2) / (*this)(system);
}

Real SumPooling::laplacian(const System& system)
{
    const int N  = system.rows();
    Real      l1 = 0;
    Real      l2 = 0;
    Real      d1 = 0;
    Real      d2 = 0;
    System    sub1(2, system.cols());
    System    sub2(2, system.cols());
    for (int i = 0; i < N; ++i)
    {
#pragma omp parallel sections
        {
#pragma omp section
            {
                sub1.row(0) = system.row(i);
                for (int j = 0; j < i; ++j)
                {
                    sub1.row(1) = system.row(j);
                    Real eval   = (*f)(sub1);
                    l1 += eval * f->laplacian(sub1);
                    d1 += eval;
                }
            }
#pragma omp section
            {
                sub2.row(0) = system.row(i);
                for (int j = i + 1; j < N; ++j)
                {
                    sub2.row(1) = system.row(j);
                    Real eval   = (*f)(sub2);
                    l2 += eval * f->laplacian(sub2);
                    d2 += eval;
                }
            }
        }
    }
    return (l1 + l2) / (d1 + d2);
}
