#pragma once
#include "activation.hpp"
#include "layer.hpp"
#include "wavefunction.hpp"

#include <memory>
#include <vector>

class Dnn : public Wavefunction
{
protected:
    unsigned                        paramCount = 0;
    std::vector<layer::DenseLayer*> layers;
    RowVector                       paramGradient;
    RowVector                       inputGradient;

    std::size_t forward_hash;

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
    Dnn() = default;

    /**
     * Append a layer to the network.
     */
    void addLayer(layer::DenseLayer& layer);

    Real operator()(const System& system) override;

    /**
     * Returns the gradient of the output w.r.t. all parameters
     * in the network, summed over the rows of x, where each
     * row of x makes up one sample input.
     */
    RowVector gradient(const System& system) override;

    Real drift_force(const System& system, int k, int dim_index) override;
    /**
     * Return the lapliacian of the output w.r.t. the inputs x
     * summed over the rows of X.
     */
    Real laplacian(const System& system) override;

    // Getters.
    const std::vector<layer::DenseLayer*>& getLayers() const;

    void set_parameters(const RowVector& parameters) override;
};

inline const std::vector<layer::DenseLayer*>& Dnn::getLayers() const
{
    return layers;
}
