import numpy as np
import scipy.stats as st
import math


def compute_statistics_for_series(x, method="plain", **method_kwargs):
    """Return statistics related to the series x."""
    supported_methods = {"plain": lambda x, **kwargs: x, "blocking": blocking}

    if method not in supported_methods:
        raise NotImplementedError(
            f"The method {method} is not supported. Available options: {supported_methods}"
        )

    method = supported_methods[method]

    x = np.asarray(x)
    x = method(x, **method_kwargs)

    return {
        "mean": np.mean(x),
        "max": np.max(x),
        "min": np.min(x),
        "var": np.var(x),
        "std": np.std(x),
        "sem": st.sem(x),
        "CI": st.t.interval(0.95, len(x) - 1, loc=np.mean(x), scale=st.sem(x)),
    }


def statistics_to_tex(
    all_stats, labels, filename=None, quantity_name="$\\langle E_L\\rangle$"
):
    """
    Produce LaTeX table from statistics produced from ``compute_statistics_for_series``.
    """

    digs = max([abs(int(math.floor(math.log10(stats["sem"])))) for stats in all_stats])

    tex = f"""\\begin{{tabular}}{{lS[table-format=1.%d]*2{{S[table-format=1.%d]}}*2{{S[table-format=1.1]}}}}
\\toprule
\\addlinespace
& {{%s}} & {{CI$^{{95}}_-$}} & {{CI$^{{95}}_+$}} & {{Std}} & {{Var}} \\\\
\\addlinespace
\\midrule
\\addlinespace
\\addlinespace
    """ % (
        digs + 2,
        digs,
        quantity_name,
    )

    for stats, label in zip(all_stats, labels):
        stats = stats.copy()
        stats["ci-"], stats["ci+"] = stats.pop("CI")

        tex += label + " & "

        significant_digits = abs(int(math.floor(math.log10(stats["sem"]))))
        tex += "{0:.{2}f}({1:.0f}) & ".format(
            stats["mean"], stats["sem"] * 10 ** significant_digits, significant_digits
        )
        tex += "{0:.{1}f} & ".format(stats["ci-"], significant_digits)
        tex += "{0:.{1}f} & ".format(stats["ci+"], significant_digits)
        tex += "\\num{{{0:.1e}}} & ".format(stats["std"])
        tex += "\\num{{{0:.1e}}}".format(stats["var"])

        tex += "\\\\\n"

    tex += "\\addlinespace\\addlinespace\\bottomrule\n\\end{tabular}"

    if filename is not None:
        with open(filename, "w") as f:
            f.write(tex)

    return tex


def blocking(x):
    """
    Return an improved estimate of the standard error
    of the mean of the given time series, accounting for
    covariant samples.
    Code by Marius Jonsson.
    Adapted by Bendik Samseth
    """
    # preliminaries
    n = len(x)
    d = int(np.log2(n))
    x_blocks = []
    s = np.empty(d)
    gamma = np.empty(d)
    mu = np.mean(x)

    assert 2 ** d == len(x)  # Size of array must be a power of two.

    # estimate the auto-covariance and variances
    # for each blocking transformation
    for i in range(0, d):
        n = len(x)
        # estimate autocovariance of x
        gamma[i] = np.sum((x[0 : (n - 1)] - mu) * (x[1:n] - mu)) / n
        # Estimate variance
        s[i] = np.var(x)
        # perform blocking transformation
        x = 0.5 * (x[0::2] + x[1::2])
        x_blocks.append(x)

    # generate the test observator M_k from the theorem
    M = (np.cumsum(((gamma / s) ** 2 * 2 ** np.arange(1, d + 1)[::-1])[::-1]))[::-1]

    # we need a list of magic numbers
    q = np.array(
        [
            6.634_897,
            9.210_340,
            11.344_867,
            13.276_704,
            15.086_272,
            16.811_894,
            18.475_307,
            20.090_235,
            21.665_994,
            23.209_251,
            24.724_970,
            26.216_967,
            27.688_250,
            29.141_238,
            30.577_914,
            31.999_927,
            33.408_664,
            34.805_306,
            36.190_869,
            37.566_235,
            38.932_173,
            40.289_360,
            41.638_398,
            42.979_820,
            44.314_105,
            45.641_683,
            46.962_942,
            48.278_236,
            49.587_884,
            50.892_181,
        ]
    )

    # use magic to determine when we should have stopped blocking
    for k in range(0, min(d, len(q))):
        if M[k] < q[k]:
            break

    if k >= d - 1 and s[k] != 0:
        raise RuntimeWarning(
            "Blocking warning: Blocked until stopped and var is not zero. You probably need more data."
        )

    # Return the new data points, corresponding to the best block size.
    return x_blocks[k]
