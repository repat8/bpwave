*******************
Signal Input/Output
*******************

.. currentmodule:: bpwave

Overview
========

By default, :class:`Signal` objects can be serialized to HDF5 using the methods
:meth:`Signal.to_hdf` and :meth:`Signal.from_hdf`.

However, conversion from other formats, such as sensor logs may be needed.

API Reference
=============

.. autoclass:: bpwave.SignalReader
    :members:
    :special-members: __call__

.. autoclass:: bpwave.CsvReader
    :members:

.. autofunction:: bpwave.to_csv
