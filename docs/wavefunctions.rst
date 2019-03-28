.. automodule:: _qflow_backend.wavefunctions

.. autoclass:: _qflow_backend.wavefunctions.Wavefunction
   :members: __call__, gradient, drift_force, laplacian, symmetry_metric

   .. autoattribute:: _qflow_backend.wavefunctions.Wavefunction.parameters

Standalone Wavefunctions
^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: _qflow_backend.wavefunctions.SimpleGaussian
   :members: __init__

.. autoclass:: _qflow_backend.wavefunctions.HardSphereWavefunction
   :members: __init__

.. autoclass:: _qflow_backend.wavefunctions.JastrowPade
   :members: __init__

.. autoclass:: _qflow_backend.wavefunctions.JastrowOrion
   :members: __init__

.. autoclass:: _qflow_backend.wavefunctions.RBMWavefunction
   :members: __init__

Composite Wavfunctions
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: _qflow_backend.wavefunctions.FixedWavefunction
   :members: __init__

.. autoclass:: _qflow_backend.wavefunctions.WavefunctionProduct
   :members: __init__

.. autoclass:: _qflow_backend.wavefunctions.SumPooling
   :members: __init__
