import numpy as np
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays


@st.composite
def position_array_shapes(draw, min_dims=1, max_dims=3, max_size=50):
    """Return a strategy for array shapes ([1...max_size], [1...3])"""
    dims = draw(st.integers(min_dims, max_dims))
    N = draw(st.integers(1, max_size))
    return (N, dims)


def float_strat():
    return st.floats(allow_nan=False, allow_infinity=False, width=16)


def array_strat(min_dims=1, max_dims=3, max_size=50):
    return arrays(
        dtype=np.float64,
        shape=position_array_shapes(
            min_dims=min_dims, max_dims=max_dims, max_size=max_size
        ),
        elements=float_strat(),
    )


def assert_close(expected, actuals, rtol=1e-05, atol=1e-08):
    assert np.shape(expected) == np.shape(
        actuals
    ), f"{np.shape(expected)} != {np.shape(actuals)}"
    expected, actuals = np.asarray(expected).ravel(), np.asarray(actuals).ravel()

    for expect, actual in zip(expected, actuals):
        if not np.isfinite(expect):
            continue

        error_msg = f"Expected {expect}, actual {actual}"

        if expect == 0:
            assert abs(actual) < atol, error_msg
        else:
            assert np.isclose(1, actual / expect, rtol=rtol, atol=atol), error_msg
