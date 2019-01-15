#include <cassert>
#include <iostream>
#include "dnn.hpp"

void Dnn::addLayer(layer::DenseLayer layer) {
    layers.push_back(layer);
    paramCount += layer.getNumberOfParameter();
    paramGradient.resize(paramCount);

    if (layers.size() == 1)
        // inputGradient has size equal to the inputs to the first layer.
        inputGradient.resize(layer.getWeights().rows());
}

const Matrix& Dnn::evaluate(const MatrixRef& x) {
    forward(x);
    return layers[layers.size() - 1].getOutputs();
}

const Vector& Dnn::parameterGradient(const MatrixRef& x) {
    forward(x);
    backward();
    unsigned k = 0;
    for (const auto& layer : layers) {
        const auto& W_grad = layer.getWeightsGradient();
        for (unsigned i = 0; i < W_grad.size(); ++i)
            paramGradient(k++) = W_grad.data()[i];
        const auto& b_grad = layer.getBiasGradient();
        for (unsigned i = 0; i < b_grad.size(); ++i)
            paramGradient(k++) = b_grad[i];
    }
    assert(k == paramCount);
    return paramGradient;
}

const Vector& Dnn::gradient(const MatrixRef& x) {
    forward(x);

    for (int j = 0; j < x.cols(); ++j) {
        Matrix dadx_j = Matrix::Zero(x.rows(), x.cols());
        dadx_j.col(j) = Matrix::Constant(x.rows(), 1, 1);
        for (auto& layer : layers) {
            dadx_j = layer.forwardGradient(dadx_j);
        }
        // Note, we assume that the output of the network is scalar.
        // If we need non-scalar outputs at some point, then this must be
        // rethinked in terms of what we want to mean by the gradient of a vector
        // output wrt. a matrix input.
        assert(dadx_j.size() == x.rows());
        inputGradient(j) = dadx_j.sum();
    }
    return inputGradient;
}

Real Dnn::laplace(const MatrixRef& x) {
    forward(x);

    Real res = 0;

    for (int j = 0; j < x.cols(); ++j) {
        Matrix ddaddx_j = Matrix::Zero(x.rows(), x.cols());
        Matrix dadx_j   = Matrix::Zero(x.rows(), x.cols());
        dadx_j.col(j)   = Matrix::Constant(x.rows(), 1, 1);

        for (auto& layer : layers) {
            ddaddx_j = layer.forwardLaplace(ddaddx_j, dadx_j);
            dadx_j = layer.forwardGradient(dadx_j);
        }
        res += ddaddx_j.sum();
    }

    return res;
}


void Dnn::forward(const MatrixRef& x) {
    assert(layers.size() > 0);

    auto layerIterator = layers.begin();
    Matrix y = layerIterator->forward(x);
    for (++layerIterator; layerIterator != layers.end(); ++layerIterator) {
        y = layerIterator->forward(y);
    }
}

void Dnn::backward() {
    assert(layers.size() > 0);
    const auto& output = layers[layers.size() - 1].getOutputs();
    Matrix y = Matrix::Ones(output.rows(), output.cols());
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        y = it->backward(y);
    }
}

