0.0.2 (2024--)
    Breaking changes:

    * ``Signal.copy``: arguments made keyword-only

    Features:

    * ``CpIndices.NAMES``
    * New attribute ``Signal.slices``

    Fixes:

    * Error message in ``Signal.chpoints`` setter.
    * Typing of ``Signal.y``.
    * Validate in ``Signal`` that ``y``, ``t`` and ``marks`` value is 1D
      (to prevent scalar values from wrongly called ``numpy`` functions).

0.0.1 (2023-01-12)
    First release.
