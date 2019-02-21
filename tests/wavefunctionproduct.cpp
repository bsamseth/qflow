#include <gtest/gtest.h>
#include <memory>

#include "definitions.hpp"
#include "system.hpp"
#include "simplegaussian.hpp"
#include "wavefunctionproduct.hpp"


TEST(WavefunctionProduct, product_of_simple_gaussians) {
    System s = System::Random(10, 3);
    SimpleGaussian psi1 (0.3);
    SimpleGaussian psi2 (0.7);
    Wavefunction* psi_squared = new SimpleGaussian(1);
    Wavefunction* psi_prod = new WavefunctionProduct(psi1, psi2);

    ASSERT_FLOAT_EQ(psi_squared->operator()(s), psi_prod->operator()(s));
    ASSERT_DOUBLE_EQ(psi_squared->laplacian(s), psi_prod->laplacian(s));
    ASSERT_TRUE(psi_squared->drift_force(s).isApprox(psi_prod->drift_force(s)));
    bool gradient_equal = Eigen::Replicate<RowVector, 1, 2>(psi_squared->gradient(s)).isApprox(psi_prod->gradient(s));
    ASSERT_TRUE(gradient_equal);

    delete psi_squared;
    delete psi_prod;
}
