*********************
Signal Representation
*********************

.. currentmodule:: bpwave

Overview
========

The package provides :class:`Signal` as a representation of continuous blood
pressure waveforms; it encapsulates data commonly needed for signal processing,
plotting and storage:

*   pressure values with unit,
*   sampling frequency,
*   timestamps,
*   characteristic points,
*   markers (arbitrary, named indices) and
*   key-value metadata.

:class:`Signal` supports serialization to HDF5 [HDF5]_ using the :mod:`h5py`
package [h5py]_ with methods :meth:`Signal.from_hdf` and :meth:`Signal.to_hdf`.

References
==========

.. [HDF5] https://www.hdfgroup.org/solutions/hdf5/
.. [h5py] https://docs.h5py.org/en/stable/index.html

API Reference
=============

.. autoclass:: bpwave.Signal
    :members:
    :class-doc-from: class
    :special-members: __init__, __getitem__

.. autoclass:: bpwave.CpIndices
    :members:
    :special-members: __sub__

.. autoclass:: bpwave.ChPoints
    :members:
