#pragma once
#include <vector>
#include "definitions.hpp"
#include "activation.hpp"
#include "layer.hpp"

class Dnn {
    protected:

        unsigned paramCount = 0;
        std::vector<layer::DenseLayer> layers;
        Vector paramGradient;
        Vector inputGradient;

        /**
         * Forward propagation of input (evaluation of network).
         */
        void forward(const MatrixRef& x);

        /**
         * Calculate derivative of output w.r.t. each component of the DNN.
         *
         * This is standard backprop. with the alteration of not propagating
         * the derivative of a cost function, but rather that of the output it self.
         * The backprop. algorithm can be used by letting the output be the cost,
         * and so the cost_gradient is simply all 1's.
         */
        void backward();

    public:

        /**
         * Append a layer to the network.
         */
        void addLayer(layer::DenseLayer layer);

        /**
         * Returns the output of the network for the input x.
         */
        const Matrix& evaluate(const MatrixRef& x);

        /**
         * Returns the gradient of the output w.r.t. all parameters
         * in the network, summed over the rows of x, where each
         * row of x makes up one sample input.
         */
        const Vector& parameterGradient(const MatrixRef& x);

        /**
         * Returns the gradient of the output w.r.t. the input X,
         * summed over the rows of X, where each row in X makes up one
         * sample input.
         */
        const Vector& gradient(const MatrixRef& x);

        /**
         * Return the lapliacian of the output w.r.t. the inputs x
         * summed over the rows of X.
         */
        Real laplace(const MatrixRef& x);

        // Getters.
        const std::vector<layer::DenseLayer>& getLayers() const;
};

inline const std::vector<layer::DenseLayer>& Dnn::getLayers() const {
    return layers;
}
