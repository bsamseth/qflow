import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

# plt.style.use('seaborn-colorblind')
# plt.style.use('grayscale')
from tqdm import tqdm


def gauss(x, mu=0, sigma=1):
    return (
        1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(-0.5 * (x - mu) ** 2 / sigma ** 2)
    )


def f(x):
    return (1 + x ** 2) ** (-0.5) * gauss(x)


if __name__ == "__main__":
    np.random.seed(2019)
    lower, upper = -5, 5
    exact = 0.78964
    V = upper - lower
    Ns = np.logspace(3, 6, 50, dtype=int)
    data = []
    for N in tqdm(Ns):
        brute = V * f(np.random.random(size=N) * V + lower)
        imp_x = np.random.normal(0, scale=1, size=N)
        imp = f(imp_x) / gauss(imp_x, 0, 1)
        data.append(
            (np.mean(brute), scipy.stats.sem(brute), np.mean(imp), scipy.stats.sem(imp))
        )
    data = np.asarray(data)
    plt.figure(figsize=(8, 8))
    plt.semilogx(Ns, data[:, 0], label="Standard Monte Carlo")
    plt.fill_between(
        Ns, data[:, 0] - 1.96 * data[:, 1], data[:, 0] + 1.96 * data[:, 1], alpha=0.2
    )
    plt.semilogx(Ns, data[:, 2], label="Importance Sampling")
    plt.fill_between(
        Ns, data[:, 2] - 1.96 * data[:, 3], data[:, 2] + 1.96 * data[:, 3], alpha=0.2
    )
    plt.hlines(exact, Ns[0], Ns[-1], linestyles="dashed")
    plt.ylim(exact - 0.05, exact + 0.1)
    plt.legend()
    plt.xlabel("Number of points")
    plt.ylabel("Integral")
    plt.legend()

    import matplotlib2tikz

    matplotlib2tikz.save(__file__ + ".tex")
    plt.show()
