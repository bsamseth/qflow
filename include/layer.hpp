#pragma once

#include "definitions.hpp"
#include "activation.hpp"

namespace layer {

class DenseLayer {
    protected:
        Matrix W;
        Matrix W_grad;
        RowVector b;
        RowVector b_grad;
        Matrix inputs;
        Matrix outputs;
        Matrix delta;
        const activation::ActivationFunction* actFunc;

    public:

        DenseLayer(int inputs, int outputs, const activation::ActivationFunction& actFunc);
        const Matrix& forward(const MatrixRef& x);
        Matrix backward(const MatrixRef& error);
        Matrix forwardGradient(const MatrixRef& dadx_j_prev);
        Matrix forwardLaplace(const MatrixRef& ddaddx_j, const MatrixRef& dadx_j);

        // Getters
        const Matrix& getOutputs() const;
        const Matrix& getWeights() const;
        const RowVector& getBiases() const;
        const Matrix& getWeightsGradient() const;
        const RowVector& getBiasGradient() const;
        unsigned getNumberOfParameter() const;
};

inline const Matrix& DenseLayer::getOutputs() const {
    return outputs;
}
inline const Matrix& DenseLayer::getWeights() const {
    return W;
}
inline const RowVector& DenseLayer::getBiases() const {
    return b;
}
inline unsigned DenseLayer::getNumberOfParameter() const {
    return W.size() + b.size();
}
inline const Matrix& DenseLayer::getWeightsGradient() const {
    return W_grad;
}
inline const RowVector& DenseLayer::getBiasGradient() const {
    return b_grad;
}

}
