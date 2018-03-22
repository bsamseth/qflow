#ifndef VMCIMPORTANCESOLVER_HPP
#define VMCIMPORTANCESOLVER_HPP

#include "vmcsolver.hpp"

namespace VMC {

class VMCImportanceSolver : public VMCSolver {
public:

    VMCImportanceSolver(const VMCConfiguration &config);

    void quantum_force(const arma::Mat<Real> &R, arma::Col<Real> &Q_force, int particle);

    virtual Results run_MC(const int n_cycles, std::ostream *out = nullptr, const double alpha = 0.5, const double beta = 1);
};

} // namespace VMC


#endif // VMCIMPORTANCESOLVER_HPP
