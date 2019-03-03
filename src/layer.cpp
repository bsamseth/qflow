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

const Matrix& InputLayer::forward(const MatrixRef& x)
{
    inputs         = x;
    Matrix z       = (inputs * V.replicate(inputs.cols(), 1)).rowwise() + b;
    return outputs = actFunc->evaluate(z);
}

Matrix InputLayer::backward(const MatrixRef& error)
{
    delta = error.cwiseProduct(actFunc->derivative(outputs));

    V_grad = (inputs.transpose() * delta).row(0);
    b_grad = delta.colwise().sum();

    // Return value does not matter since this is the input layer.
    // No one to send an error back to.
    return Matrix::Zero(0, 0);
}

Matrix InputLayer::forwardGradient(const MatrixRef& dadx_j)
{
    return actFunc->derivative(outputs).cwiseProduct(dadx_j
                                                     * V.replicate(dadx_j.cols(), 1));
}

Matrix InputLayer::forwardLaplace(const MatrixRef& ddaddx_j, const MatrixRef& dadx_j)
{
    // For some reason, Eigen needs to evaluate both parts as matrices completely
    // before they can be summed. Letting 'first' be auto we get wrong answers, while
    // letting second be auto does not compile. Strange, but this way works!
    Matrix first = actFunc->derivative(outputs).cwiseProduct(
        ddaddx_j * V.replicate(ddaddx_j.cols(), 1));
    Matrix second = actFunc->dblDerivative(outputs).array()
                    * (dadx_j * V.replicate(dadx_j.cols(), 1)).array().square();
    return first + second;
}

}  // namespace layer
