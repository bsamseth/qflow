#ifndef VMCIMPORTANCESOLVER_HPP
#define VMCIMPORTANCESOLVER_HPP

#include "vmcsolver.hpp"

namespace VMC {

class VMCImportanceSolver : public VMCSolver {
public:

    VMCImportanceSolver(const VMCConfiguration &config);

    void quantum_force(const arma::Mat<Real> &R, arma::Col<Real> &Q_force, int particle);

    Results run_MC(const int n_cycles);
};

} // namespace VMC


#endif // VMCIMPORTANCESOLVER_HPP
