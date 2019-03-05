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

//
// Boilerplate getter implementations.
//
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

}  // namespace layer
