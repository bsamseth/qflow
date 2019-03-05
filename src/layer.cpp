#include "layer.hpp"

#include <cmath>
#include <iostream>

namespace layer
{
DenseLayer::DenseLayer(int                               inputs,
                       int                               outputs,
                       const activation::ReluActivation& actFunc)
    : DenseLayer(inputs, outputs, actFunc, std::sqrt(2.0 / inputs))
{
}

DenseLayer::DenseLayer(int                               inputs,
                       int                               outputs,
                       const activation::TanhActivation& actFunc)
    : DenseLayer(inputs, outputs, actFunc, std::sqrt(1.0 / inputs))
{
}

DenseLayer::DenseLayer(int                                   inputs,
                       int                                   outputs,
                       const activation::ActivationFunction& actFunc)
    : DenseLayer(inputs, outputs, actFunc, std::sqrt(2.0 / (inputs + outputs)))
{
}

DenseLayer::DenseLayer(int                                   inputs,
                       int                                   outputs,
                       const activation::ActivationFunction& actFunc,
                       Real                                  scale_factor)
    : W(Matrix::Zero(inputs, outputs))
    , W_grad(W)
    , b(RowVector::Zero(outputs))
    , b_grad(b)
    , actFunc(&actFunc)
{
    for (int i = 0; i < inputs; ++i)
    {
        for (int j = 0; j < outputs; ++j)
        {
            W(i, j) = rnorm_func();
        }
    }
    W *= scale_factor;
}

const Matrix& DenseLayer::forward(const MatrixRef& x)
{
    inputs         = x;
    Matrix z       = (inputs * W).rowwise() + b;
    return outputs = actFunc->evaluate(z);
}

Matrix DenseLayer::backward(const MatrixRef& error)
{
    delta = error.cwiseProduct(actFunc->derivative(outputs));

    W_grad = inputs.transpose() * delta;
    b_grad = delta.colwise().sum();

    return delta * W.transpose();
}

Matrix DenseLayer::forwardGradient(const MatrixRef& dadx_j)
{
    return actFunc->derivative(outputs).cwiseProduct(dadx_j * W);
}

Matrix DenseLayer::forwardLaplace(const MatrixRef& ddaddx_j, const MatrixRef& dadx_j)
{
    // For some reason, Eigen needs to evaluate both parts as matrices completely
    // before they can be summed. Letting 'first' be auto we get wrong answers, while
    // letting second be auto does not compile. Strange, but this way works!
    Matrix first = actFunc->derivative(outputs).cwiseProduct(ddaddx_j * W);
    Matrix second
        = actFunc->dblDerivative(outputs).array() * (dadx_j * W).array().square();
    return first + second;
}

}  // namespace layer
