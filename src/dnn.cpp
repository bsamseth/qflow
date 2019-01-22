#include <cassert>
#include <iostream>
#include "dnn.hpp"

void Dnn::addLayer(const layer::DenseLayer& layer) {
    layers.push_back(layer);
    paramCount += layer.getNumberOfParameter();
    paramGradient.resize(paramCount);
    _parameters.resize(paramCount);

    // Copy parameters from layers.
    unsigned k = 0;
    for (auto& layer : layers) {
        auto& W = layer.getWeights();
        auto& b = layer.getBiases();
        for (unsigned i = 0; i < W.size(); ++i)
            _parameters[k++] = W.data()[i];
        for (unsigned i = 0; i < b.size(); ++i)
            _parameters[k++] = b[i];
    }
    assert(k == paramCount);

    if (layers.size() == 1)
        // inputGradient has size equal to the inputs to the first layer.
        inputGradient.resize(layer.getWeights().rows());
}

Real Dnn::operator()(System& system) {
    assert(layers[layers.size() - 1].getWeights().cols() == 1);  // Output layer should have only one output.

    Eigen::Map<RowVector> x(system.data(), system.size());  // Reshape as (1 x n) matrix
    forward(x);

    return layers[layers.size() - 1].getOutputs()(0,0);
}

RowVector Dnn::gradient(System& system) {
    Real output = (*this)(system);
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
    return paramGradient / output;
}

const RowVector& Dnn::positionGradient(System& system) {
    assert(layers[layers.size() - 1].getWeights().cols() == 1);  // Output layer should have only one output.

    Eigen::Map<RowVector> x(system.data(), system.size());  // Reshape as (1 x n) matrix
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

Real Dnn::laplacian(System& system) {
    assert(layers[layers.size() - 1].getWeights().cols() == 1);  // Output layer should have only one output.

    Eigen::Map<RowVector> x(system.data(), system.size());  // Reshape as (1 x n) matrix
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

    return res / layers[layers.size() - 1].getOutputs()(0,0);;
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

void Dnn::set_parameters(const RowVector& parameters) {
    _parameters = parameters;

    // Fill in layers.
    unsigned k = 0;
    for (auto& layer : layers) {
        auto& W = layer.getWeights();
        auto& b = layer.getBiases();
        for (unsigned i = 0; i < W.size(); ++i)
            W.data()[i] = parameters[k++];
        for (unsigned i = 0; i < b.size(); ++i)
            b[i] = parameters[k++];
    }
    assert(k == paramCount);
}
