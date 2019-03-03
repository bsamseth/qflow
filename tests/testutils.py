import numpy as np
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays

float_strat = lambda: st.floats(allow_nan=False, allow_infinity=False, width=32)
array_strat = lambda min_size=1, max_size=10: arrays(
    dtype=np.float64,
    shape=array_shapes(min_dims=2, max_dims=2, min_side=min_size, max_side=max_size),
    elements=float_strat(),
)


def assert_close(expected, actuals, rtol=1e-05, atol=1e-08):
    assert np.shape(expected) == np.shape(actuals)
    expected, actuals = np.asarray(expected).ravel(), np.asarray(actuals).ravel()

    for expect, actual in zip(expected, actuals):
        if not np.isfinite(expect):
            continue

        if expect == 0:
            assert expect == actual
        else:
            assert np.isclose(1, actual / expect, rtol=rtol, atol=atol)
