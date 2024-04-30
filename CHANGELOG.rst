0.0.3 (2024-04-30)
    Features:

    * ``SignalReader`` base class and ``CsvReader``.
    * ``Signal.plot``: grid on by default. New parameter ``t0``.
    * ``to_csv``: simple CSV export
    * ``ChPoints.plot``: plot independently of the containing signal.
    * Slicing of ``ChPoints``, shifting indices with scalar (``+, -``)
    * ``visu.plot_signal_slices``.

    Fixes:

    * ``Signal.__getitem__`` can no longer produce empty slices in ``Signal.slices``.
    * ``Signal.slices`` validation error messages now contain the ``key``.

0.0.2 (2024-02-27)
    Breaking changes:

    * ``Signal.copy``: arguments made keyword-only

    Features:

    * ``CpIndices.NAMES``
    * New attribute ``Signal.slices``
    * Slicing ``Signal`` by time or onsets: ``by_t``, ``by_onset``.
    * ``Signal.t2i`` accepts backward (negative) time point.

    Fixes:

    * Error message in ``Signal.chpoints`` setter.
    * Typing of ``Signal.y``.
    * Validate in ``Signal`` that ``y``, ``t`` and ``marks`` value is 1D
      (to prevent scalar values from wrongly called ``numpy`` functions).

0.0.1 (2023-01-12)
    First release.
