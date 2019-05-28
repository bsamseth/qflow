import numpy as np
from itertools import permutations
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from qflow.wavefunctions.nn.layers import DenseLayer
from qflow.wavefunctions import Dnn, InputSorter


@given(P=st.integers(1, 20), D=st.integers(1, 3))
def test_symmetry(P, D):
    dnn = Dnn()
    layer = DenseLayer(P * D, 1)
    dnn.add_layer(layer)

    sdnn = InputSorter(dnn)

    # Random system, sorted by distance to origin.
    s = np.random.randn(P, D)
    s = s[np.argsort([row.dot(row) for row in s])]

    # Limit the number of permutations, for computational feasibility.
    for _, p in zip(range(100), permutations(s)):
        p = np.asarray(p)

        # Permutation invariant:
        assert sdnn(s) == sdnn(p)

        # Values should be equal to dnn applied to the pre-sorted system.
        assert sdnn(p) == dnn(s)
        assert sdnn.laplacian(p) == dnn.laplacian(s)
        assert all(sdnn.gradient(p) == dnn.gradient(s))

        # Drift force should have equal values, but permuted according to the permutation p.
        # This because the drift force depends on which positions are which. Values should
        # still be the same, so we only allow for sequence particle-wise permutations.
        sdnn_drift, dnn_drift = (
            sdnn.drift_force(p).reshape(s.shape),
            dnn.drift_force(s).reshape(s.shape),
        )
        sdnn_drift.sort(axis=0)
        dnn_drift.sort(axis=0)
        assert (sdnn_drift == dnn_drift).all()
