#include "wavefunctionproduct.hpp"

WavefunctionProduct::WavefunctionProduct(Wavefunction& a, Wavefunction& b)
    : f(&a), g(&b)
{
    _parameters.resize(f->get_parameters().size() + g->get_parameters().size());
    _parameters << f->get_parameters(), g->get_parameters();
}

void WavefunctionProduct::set_parameters(const RowVector& a, const RowVector& b)
{
    f->set_parameters(a);
    g->set_parameters(b);
    _parameters << f->get_parameters(), g->get_parameters();
}

void WavefunctionProduct::set_parameters(const RowVector& parameters)
{
    Eigen::Map<const RowVector> a(parameters.data(), f->get_parameters().size());
    Eigen::Map<const RowVector> b(parameters.data() + f->get_parameters().size(),
                                  g->get_parameters().size());
    set_parameters(a, b);
}

RowVector WavefunctionProduct::gradient(const System& system)
{
    RowVector grad_a = f->gradient(system);
    RowVector grad_b = g->gradient(system);
    RowVector res(grad_a.size() + grad_b.size());
    res << grad_a, grad_b;
    return res;
}
