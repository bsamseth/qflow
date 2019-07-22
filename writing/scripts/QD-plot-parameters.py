import numpy as np
import matplotlib.pyplot as plt
import matplotlib2tikz

parameters = [
    np.loadtxt("QD-parameters-dnn-regular.txt"),
    np.loadtxt("QD-parameters-dnn-sorted.txt"),
]


for name, p in zip(["regular", "sorted"], parameters):
    fig, axes = plt.subplots(ncols=2, nrows=3)
    i, j = 4, 4 + 4 * 32
    w0, b0 = p[i:j], p[j : j + 32]
    i, j = j + 32, j + 32 + 32 * 16
    w1, b1 = p[i:j], p[j : j + 16]
    i, j = j + 16, j + 16 + 16
    w2, b2 = p[i:j], p[j : j + 1]

    axes[0][0].matshow(np.reshape(w0, (4, 32)))
    axes[1][0].matshow(np.reshape(w1, (32, 16)), aspect="auto")
    axes[2][0].matshow(np.reshape(w2, (16, 1)), aspect="auto")

    axes[0][1].matshow(np.reshape(b0, (1, -1)), aspect="auto")
    axes[1][1].matshow(np.reshape(b1, (1, -1)), aspect="auto")
    axes[2][1].matshow(np.reshape(b2, (1, -1)), aspect="auto")

    axes[0][0].set_ylabel(r"$\mathbf{W}^{(0)}$")
    axes[1][0].set_ylabel(r"$\mathbf{W}^{(1)}$")
    axes[2][0].set_ylabel(r"$\mathbf{W}^{(2)}$")
    axes[2][0].set_xlabel("Layer weights")

    axes[0][1].set_ylabel(r"$\mathbf{b}^{(0)}$")
    axes[1][1].set_ylabel(r"$\mathbf{b}^{(1)}$")
    axes[2][1].set_ylabel(r"$\mathbf{b}^{(2)}$")
    axes[2][1].set_xlabel("Layer biases")

    axes[0][1].yaxis.set_label_position("right")
    axes[1][1].yaxis.set_label_position("right")
    axes[2][1].yaxis.set_label_position("right")
    fig.tight_layout()

    matplotlib2tikz.save(
        __file__ + f".weights-{name}.tex", tex_relative_path_to_data="scripts/"
    )

plt.show()
