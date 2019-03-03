#pragma once

#include "activation.hpp"
#include "definitions.hpp"

namespace layer
{
class DenseLayer
{
protected:
    Matrix                                      W;
    Matrix                                      W_grad;
    RowVector                                   b;
    RowVector                                   b_grad;
    Matrix                                      inputs;
    Matrix                                      outputs;
    Matrix                                      delta;
    const activation::ActivationFunction* const actFunc;

public:
    DenseLayer(int                                   inputs,
               int                                   outputs,
               const activation::ActivationFunction& actFunc,
               Real                                  scale_factor);
    DenseLayer(int inputs, int outputs, const activation::ReluActivation& actFunc);
    DenseLayer(int inputs, int outputs, const activation::TanhActivation& actFunc);
    DenseLayer(int inputs, int outputs, const activation::ActivationFunction& actFunc);
    virtual const Matrix& forward(const MatrixRef& x);
    virtual Matrix        backward(const MatrixRef& error);
    virtual Matrix        forwardGradient(const MatrixRef& dadx_j_prev);
    virtual Matrix forwardLaplace(const MatrixRef& ddaddx_j, const MatrixRef& dadx_j);

    // Getters
    const Matrix&         getOutputs() const;
    virtual const Matrix& getWeights() const;
    virtual Matrix&       getWeights();
    const RowVector&      getBiases() const;
    RowVector&            getBiases();
    virtual const Matrix& getWeightsGradient() const;
    const RowVector&      getBiasGradient() const;
    virtual unsigned      getNumberOfParameter() const;
};

/**
 * Special layer used only as the first layer in a Wavefunction Dnn.
 * This will enforce that the Dnn is **symmetric** on any permutation of
 * the inputs, which is a requirement for valid wavefunctions.
 *
 * This acts in all ways like a regular DenseLayer, except that the weights
 * are equivalent to a matrix with constant columns (all rows are the same).
 *
 * Note that for fermions the Dnn **must** be used as part of a WavefunctionProduct
 * where the other factor is anti-symmetric (i.e. a Slater determinant). For bosons
 * the Dnn _can_ be used alone, but better together with some symmetric ideal-case
 * wavefunction permanent.
 */
class InputLayer : public DenseLayer
{
private:
    Matrix V;
    Matrix V_grad;

public:
    // Constructors simply forward to DenseLayer + initialized V and V_grad from W and
    // W_grad.
    InputLayer(int                                   inputs,
               int                                   outputs,
               const activation::ActivationFunction& actFunc,
               Real                                  scale_factor)
        : DenseLayer(inputs, outputs, actFunc, scale_factor)
        , V(W.row(0))
        , V_grad(W_grad.row(0))
    {
    }
    InputLayer(int inputs, int outputs, const activation::ReluActivation& actFunc)
        : DenseLayer(inputs, outputs, actFunc), V(W.row(0)), V_grad(W_grad.row(0))
    {
    }
    InputLayer(int inputs, int outputs, const activation::TanhActivation& actFunc)
        : DenseLayer(inputs, outputs, actFunc), V(W.row(0)), V_grad(W_grad.row(0))
    {
    }
    InputLayer(int inputs, int outputs, const activation::ActivationFunction& actFunc)
        : DenseLayer(inputs, outputs, actFunc), V(W.row(0)), V_grad(W_grad.row(0))
    {
    }

    // Overridden functions.
    const Matrix& forward(const MatrixRef& x) override;
    Matrix        backward(const MatrixRef& error) override;
    Matrix        forwardGradient(const MatrixRef& dadx_j_prev) override;
    Matrix forwardLaplace(const MatrixRef& ddaddx_j, const MatrixRef& dadx_j) override;
    const Matrix& getWeights() const override;
    Matrix&       getWeights() override;
    const Matrix& getWeightsGradient() const override;
    unsigned      getNumberOfParameter() const override;
};

//
// Boilerplate getter implementations.
//

// DenseLayer
inline const Matrix& DenseLayer::getOutputs() const
{
    return outputs;
}
inline const Matrix& DenseLayer::getWeights() const
{
    return W;
}
inline Matrix& DenseLayer::getWeights()
{
    return W;
}
inline const RowVector& DenseLayer::getBiases() const
{
    return b;
}
inline RowVector& DenseLayer::getBiases()
{
    return b;
}
inline unsigned DenseLayer::getNumberOfParameter() const
{
    return W.size() + b.size();
}
inline const Matrix& DenseLayer::getWeightsGradient() const
{
    return W_grad;
}
inline const RowVector& DenseLayer::getBiasGradient() const
{
    return b_grad;
}

// InputLayer
inline const Matrix& InputLayer::getWeights() const
{
    return V;
}
inline Matrix& InputLayer::getWeights()
{
    return V;
}
inline const Matrix& InputLayer::getWeightsGradient() const
{
    return V_grad;
}
inline unsigned InputLayer::getNumberOfParameter() const
{
    return V.size() + b.size();
}

}  // namespace layer
