"""BP time series."""
import collections as _col
import collections.abc as _ca
import copy as _cp
import dataclasses as _dc
import json
import typing as _t
import warnings

import h5py  # type: ignore
import matplotlib.pyplot as _plt
import numpy as _np
import numpy.typing as _npt

from . import __version__
from . import visu as _v


@_dc.dataclass(kw_only=True, frozen=True)
class CpIndices:
    """Indices of the characteristic points of a single period of an ABP signal.

    Fields are optional; missing entries get negative value.

    .. warning::
        Please don't use negative (backwards) indexing, set values must be >= 0.
    """

    NAMES: _t.ClassVar[tuple[str, str, str, str, str, str]] = (
        "onset",
        "sys_peak",
        "refl_onset",
        "refl_peak",
        "dicr_notch",
        "dicr_peak",
    )
    """Index names in the same order as values in :meth:`to_array`."""

    UNSET: _t.ClassVar[int] = -1
    """Value representing unset value.

    .. versionadded:: 0.0.3
    """

    onset: int = UNSET
    """Onset."""

    sys_peak: int = UNSET
    """Systolic peak."""

    refl_onset: int = UNSET
    """Onset of the reflected wave."""

    refl_peak: int = UNSET
    """Peak of the reflected wave."""

    dicr_notch: int = UNSET
    """Dicrotic notch."""

    dicr_peak: int = UNSET
    """Peak of the dicrotic wave."""

    def __post_init__(self):
        if (self.to_array() == self.UNSET).all():
            warnings.warn("None of the indices are set.")

    def __add__(self, shift: int) -> "CpIndices":
        """Adds the same scalar to all indices.

        A possible use case is when characteristic points are detected on a
        shorter section of a signal, but the result needs to be trasferred to
        the full length one.

        .. versionadded:: 0.0.3
        """
        return CpIndices(
            **{
                n: shifted if (shifted := v + shift) >= 0 else self.UNSET
                for n, v in self.without_unset().items()
            }
        )

    def __sub__(self, shift: int) -> "CpIndices":
        """Subtracts the same scalar from all indices.

        .. versionadded:: 0.0.3
        """
        return self + -shift

    def min(self) -> int:
        """Smallest set index."""
        return min(self.without_unset().values(), default=self.UNSET)

    def max(self) -> int:
        """Largest set index."""
        return max(self.without_unset().values(), default=self.UNSET)

    def clamped(self, start: int | None = None, stop: int | None = None) -> "CpIndices":
        """Keeps only the indices contained in the specified range."""
        start = start or 0
        stop = _np.iinfo(int).max if stop is None else stop
        return CpIndices(
            **{
                n: v if start <= v < stop else self.UNSET
                for n, v in self.without_unset().items()
            }
        )

    def without_unset(self) -> dict[str, int]:
        """Returns the existing points in a ``dict``."""
        return {
            name: value for name, value in self.__dict__.items() if value != self.UNSET
        }

    def to_array(self) -> _npt.NDArray[_np.int64]:
        """The points as numpy array."""
        return _np.array(_dc.astuple(self))


@_dc.dataclass(kw_only=True, frozen=True)
class ChPoints:
    """Holds the result of a characteristic point detection algorithm."""

    indices: list[CpIndices]
    """Absolute indices of the characteristic points."""

    alg: str
    """(Qualified) name of the algorithm or ``"manual"`` if annotated by human."""

    version: str
    """Version of the algorithm's package or ISO date in case of human."""

    params: dict[str, _t.Any]
    """Parameters of the algorithm, if any."""

    def __getitem__(self, slc: slice) -> "ChPoints":
        """Returns a copy with sliced ``indices``.

        Negative (backwards) indices and setting ``step`` is not supported.

        .. versionadded:: 0.0.3
        """
        if not isinstance(slc, slice):
            raise NotImplementedError("Only slicing is supported")
        if slc.step is not None:
            raise NotImplementedError("Step is not yet supported")
        if (slc.start is not None and slc.start < 0) or (
            slc.stop is not None and slc.stop < 0
        ):
            raise NotImplementedError("Negative indices not supported")

        start = slc.start or 0
        stop = _np.iinfo(int).max if slc.stop is None else slc.stop

        return _dc.replace(
            self,
            indices=[
                ci.clamped(start, stop) - start
                for ci in self.indices
                if (
                    (ci.min() >= start and ci.max() < stop)
                    or (ci.min() <= start <= ci.max())
                    or (ci.min() < stop <= ci.max())
                )
            ],
        )

    def __add__(self, shift: int) -> "ChPoints":
        """Adds the same scalar to all indices.

        A possible use case is when characteristic points are detected on a
        shorter section of a signal, but the result needs to be trasferred to
        the full length one.

        .. versionadded:: 0.0.3
        """
        return _dc.replace(
            self,
            indices=[ci + shift for ci in self.indices],
        )

    def __sub__(self, shift: int) -> "ChPoints":
        """Subtracts the same scalar from all indices.

        .. versionadded:: 0.0.3
        """
        return self + -shift

    def plot(
        self,
        ax: _plt.Axes,
        *,
        t: _np.ndarray,
        y: _np.ndarray,
        points: _t.Literal[True] | set[str] = True,
    ) -> _plt.Axes:
        """Plots the points on the given axes ``ax`` using timestamps ``t`` and
        amplitudes ``y``.

        The purpose of this method is to be able to plot characteristic points
        independently of the containing signal, useful for previewing results
        of a detection algorithm.

        For plotting the own characteristic points of a signal, please use
        :meth:`Signal.plot`.

        .. warning::
            This method doesn't validate ``t`` and ``y`` against the indices.

        :param ax: subplot axes.
        :param t: timestamps.
        :param y: amplitudes.
        :param points: set to filter points by name. ``True`` means all.
        :return: the axes.

        .. versionadded:: 0.0.3
        """
        all_indices = _col.defaultdict(list)
        for ci in self.indices:
            for name, index in ci.without_unset().items():
                if points is True or name in points:
                    all_indices[name].append(index)
        for name, indices in all_indices.items():
            ax.plot(t[indices], y[indices], "+", label=name)
        return ax


class _MarksProxy(_col.UserDict):
    def __init__(
        self,
        markers: _t.Union[dict[str, _ca.Iterable[int]], "_MarksProxy"],
        stop: int,
    ):
        self._stop = stop
        super().__init__(markers)

    def __setitem__(self, key: str, value: _ca.Iterable[int]) -> None:
        indices = _np.asarray(value, int)
        _validate_1d(indices, "value")
        if ((indices < 0) | (indices >= self._stop)).any():
            raise ValueError(f"`{key}` must be valid nonnegative indices of `y`")
        super().__setitem__(key, indices)


_SlicesInput: _t.TypeAlias = _t.Union[dict[str, _ca.Sequence[slice]], "_SlicesProxy"]


class _SlicesProxy(_col.UserDict):
    def __init__(
        self,
        slices: _SlicesInput,
        stop: int,
    ):
        self._stop = stop
        super().__init__(slices)

    def __setitem__(self, key: str, value: _ca.Sequence[slice]) -> None:
        for i, slc in enumerate(value):
            if slc.step is not None:
                warnings.warn(f"Handling `step` is not implemented (value[{key}][{i}])")
            if slc.start < 0 or slc.start >= self._stop:
                raise ValueError(
                    f"`value[{key}][{i}].start must be valid nonnegative index of `y`"
                )
            if slc.stop < 0 or slc.stop > self._stop:
                raise ValueError(
                    f"`value[{key}][{i}].stop must be valid nonnegative index of `y`"
                )
        super().__setitem__(key, list(value))


class _SignalIndexer:
    def __init__(self, signal: "Signal"):
        self._signal = signal


class _CCyclesIndexer(_SignalIndexer):
    @_t.overload
    def __getitem__(self, item: int) -> "Signal":
        ...

    @_t.overload
    def __getitem__(self, item: slice) -> list["Signal"]:
        ...

    def __getitem__(self, item: int | slice) -> _t.Union["Signal", list["Signal"]]:
        slices = list(self._signal.iter_ccycle_slices())
        match item:
            case int():
                return self._signal[slices[item]]
            case slice():
                return [self._signal[s] for s in slices[item]]


class _ByTIndexer(_SignalIndexer):
    def __getitem__(self, item: slice) -> "Signal":
        match item:
            case slice(start=start, stop=stop, step=None):
                if start is not None:
                    start = self._signal.t2i(start)
                if stop is not None:
                    stop = self._signal.t2i(stop)
                return self._signal[start:stop]
            case _:
                raise ValueError(f"Unsupported time-based slice {item}")


class _InclusiveByOnsetIndexer(_SignalIndexer):
    def __getitem__(self, item: slice) -> "Signal":
        match item:
            case slice(start=start, stop=stop, step=None):
                if start is None:
                    start = 0
                if stop is None:
                    stop = -1
                return self._signal[
                    self._signal.onsets[start] : self._signal.onsets[stop] + 1
                ]
            case _:
                raise ValueError(f"Unsupported onset slice {item}")


class Signal:
    """Time series wrapper providing sampling frequency and time points,
    facilitating consistency, plotting, slicing and storage.

    A signal can be created either directly, e. g.::

        s = Signal(y=[94.03 , 94.46, 94.90], unit='mmHg', fs=1000.)
        s = Signal(y=[94.03 , 94.46, 94.90], unit='mmHg', t=[0., 0.001, 0.002])

    or from a HDF5 group filled by :meth:`to_hdf`::

        with h5py.File('path.hdf5', 'w') as hdf_file:
            s.to_hdf(hdf_file.create_group('measurements/01'))
        with h5py.File('path.hdf5', 'r') as hdf_file:
            s = Signal.from_hdf(hdf_file['measurements/01'])
    """

    HDF_FORMAT_VERSION: _t.Final[int] = 1
    """Version of serialization."""

    def __init__(
        self,
        y: _ca.Iterable,
        unit: str = "rel",
        *,
        t: _ca.Iterable | None = None,
        fs: float | None = None,
        label: str | None = None,
        chpoints: ChPoints | None = None,
        marks: dict[str, _ca.Iterable[int]] | _MarksProxy | None = None,
        slices: _SlicesInput | None = None,
        meta: dict[str, _t.Any] | None = None,
    ):
        """Creates a signal defined by data points and timestamps or sampling frequency.

        :param y: signal values (see :attr:`y`). Its length will determine
            the accepted length for :attr:`t`.
        :param unit: signal unit. Optional (default: ``"rel"``),
            identical to setting :attr:`unit`.
        :param t: timestamps [s] (see :attr:`t`).
            Validated for length and strict monotonicity.
        :param fs: sampling frequency [Hz] (see :attr:`fs`).
        :param label: label for plotting.
            Optional, identical to setting :attr:`label`.
        :param chpoints: characteristic points.
            Optional, identical to setting  :attr:`chpoints`.
        :param marks: named indices.
            Optional, identical to setting  :attr:`marks`.
        :param slices: named slice sequences.
            Optional, identical to setting :attr:`slices`.
        :param meta: key-value metadata.
            Optional, identical to setting  :attr:`meta`.

        .. note::
            Exactly one of ``t`` or ``fs`` must be given, the other one will be
            calculated (on first usage).

        .. warning::
            Avoid in-place modification of fields with a mutable type, as it
            will not trigger validations.

        :raises ValueError: when parameter validation failed.
        """
        if (t is None and fs is None) or (t is not None and fs is not None):
            raise ValueError("Exactly one of `t` or `fs` must be given")

        self.unit = unit  #: Unit of the signal values ``y``.
        self.label = label  #: Name of the signal (e. g. for plot label).

        self._y = _np.asarray(y, dtype=float)
        _validate_1d(self._y, "y")
        self._t: _npt.NDArray[_np.float64] | None = None

        if t is not None:
            self._t = _np.asarray(t, dtype=float)
            _validate_1d(self._t, "t")
            self._validate_timestamps()
            self._fs = 1.0 / _np.diff(self._t).mean()
            self._t_from_fs = False
        else:
            self._t = None
            self._fs = fs
            self._t_from_fs = True

        self._chpoints: ChPoints | None = None
        self._onsets = _np.array([], int)
        self.chpoints = chpoints

        self._marks = _MarksProxy({}, len(self._y))
        self.marks = marks  # type: ignore[assignment]

        self._slices = _SlicesProxy({}, len(self._y))
        self.slices = slices  # type: ignore[assignment]

        #: Key-value metadata (value can be anything accepted by
        #: :attr:`h5py.Group.attrs`.
        self.meta: dict[str, _t.Any] = meta or {}

    @property
    def y(self) -> _npt.NDArray[_np.float64]:
        """Signal values (*y* axis) in arbitrary unit :attr:`unit`.

        Can be updated with an array of values convertible to ``np.float64``,
        having the same size as the array passed to the ``y`` parameter of the
        constructor.

        :raises ValueError: when new array has different size.
        """
        return self._y

    @y.setter
    def y(self, new_y: _ca.Iterable) -> None:
        new_ya = _np.asarray(new_y, dtype=float)
        _validate_1d(new_ya, "new_y")
        if len(new_ya) != len(self._y):
            raise ValueError(f"New `y` must not change length ({len(self._y)})")
        self._y = new_ya

    @property
    def t(self) -> _npt.NDArray[_np.float64]:
        """Timestamps (*x* axis values) in seconds (readonly)."""
        if self._t is None:
            self._t = _np.arange(len(self.y)) / self._fs
        return self._t

    @property
    def t_from_fs(self) -> bool:
        """Whether timestamps were calculated from sampling frequency, i. e.
        ``t`` parameter of the constructor was not provided (readonly)."""
        return self._t_from_fs

    @property
    def fs(self) -> float:
        """Sampling frequency in Hz (readonly).

        If not provided in the constructor, calculated from the mean difference
        of consecutive timestamps.
        """
        return self._fs

    @property
    def chpoints(self) -> ChPoints | None:
        """Characteristic points by cardiac cycle with algorithm metadata.

        When set, indices are validated for range and index sets are sorted.

        :raises ValueError: when an index is out of range.
        """
        return self._chpoints

    @chpoints.setter
    def chpoints(self, s: ChPoints | None) -> None:
        if s is None:
            self._chpoints = None
            self._onsets = _np.array([], int)
            return

        i_max = len(self.y) - 1
        n_ccycles = len(s.indices)
        for i, ci in enumerate(s.indices):
            curr_i_max = i_max if i == n_ccycles - 1 else s.indices[i + 1].onset - 1
            for pname, value in ci.without_unset().items():
                if not ((0 if pname == "onset" else ci.onset) <= value <= curr_i_max):
                    raise ValueError(f"indices[{i}].{pname}={value} is out of range")
        self._chpoints = _dc.replace(
            s, indices=sorted(s.indices, key=lambda e: e.onset)
        )
        self._onsets = _np.array(
            [ci.onset for ci in self._chpoints.indices if ci.onset >= 0]
        )

    @property
    def onsets(self) -> _npt.NDArray[_np.int64]:
        """Indices of onset points, empty array if not yet stored (readonly).

        Can be modified via :attr:`chpoints`.

        .. seealso::
            :meth:`by_onset`
            :meth:`ccycles`
            :meth:`iter_ccycles`
            :meth:`iter_ccycle_slices`
        """
        return self._onsets

    @property
    def t_onsets(self) -> _npt.NDArray[_np.float64]:
        """Timestamps of onset points, empty array if not yet stored (readonly)."""
        return self.t[self._onsets]

    @property
    def marks(self) -> _MarksProxy:
        """Named index lists e. g. to mark events during the measurement.

        Index range is validated on assignments like ``signal.marks = {'a': [1]}``
        or ``signal.marks['a'] = [1]``.

        .. note::
            *mypy* gives a false positive error on assignment, use
            ``# type: ignore[assignment]``.

        :returns: an object behaving exactly like a ``dict[str, np.ndarray[int]]``.
        :raises ValueError: when a mark index is out of range
        """
        return self._marks

    @marks.setter
    def marks(self, m: dict[str, _ca.Iterable[int]] | _MarksProxy | None):
        self._marks = _MarksProxy(m or {}, len(self._y))

    @property
    def slices(self) -> _SlicesProxy:
        """Named slice lists e. g. to mark sections of the measurement.

        Index range is validated on assignments like::

            signal.slices = {'hand_movement': [np.s_[100:600]]}
            signal.slices['hand_movement'] = [slice(100, 200)]

        ``step`` is not supported in slices.

        .. note::
            *mypy* gives a false positive error on assignment, use
            ``# type: ignore[assignment]``.

        :returns: an object behaving exactly like a ``dict[str, list[slice]]``.
        :raises ValueError: when a ``start`` or ``stop`` is out of range.
        """
        return self._slices

    @slices.setter
    def slices(self, v: _SlicesInput | None):
        self._slices = _SlicesProxy(v or {}, len(self._y))

    def __getitem__(self, slc: slice) -> "Signal":
        """Creates a **view** of a section of the signal.

        Indices in :attr:`chpoints` and :attr:`marks` are shifted and only those
        are included that fall into the selected range.
        That is, the first :class:`CpIndices` object will be the first one for which
        ``onset >= slc.start`` holds and those points in the last included
        :class:`CpIndices`, that are ``>= slc.stop`` will be set to missing entries.

        When all of the indices in a :attr:`marks` entry are filtered out,
        the key will be still present with an empty array.

        .. note::
            Only range selection is supported, without step.
            That is, ``signal[:100]``, ``signal[100:]`` and ``signal[100:1000]``
            are valid, ``signal[0]`` and ``signal[100:1000:2]`` are not.

        :raises NotImplementedError: for unsopported slicing,
        """
        if not isinstance(slc, slice):
            raise NotImplementedError("Only slicing is supported")
        if slc.step is not None:
            raise NotImplementedError("Step is not yet supported")

        length = len(self.y)

        if slc.start is None:
            start = 0
        else:
            start = slc.start if slc.start >= 0 else length + slc.start

        if slc.stop is None:
            stop = length
        else:
            stop = slc.stop if slc.stop >= 0 else length + slc.stop

        def crop_slices(slcs) -> list[slice]:
            cropped = [
                slice(max(0, s.start - start), min(s.stop - start, stop - start))
                for s in slcs
                if start <= s.start < stop or start <= s.stop <= stop
            ]
            # Remove empty slices
            cropped = [s for s in cropped if s.stop > s.start]
            return cropped

        section = type(self)(
            y=self.y[slc],
            unit=self.unit,
            t=self.t[slc],  # intentionally the t property
            fs=None,
            label=self.label,
            chpoints=(None if self.chpoints is None else self.chpoints[start:stop]),
            marks={
                name: v[(v >= start) & (v < stop)] - start
                for name, v in self.marks.items()
            },
            slices={name: crop_slices(slcs) for name, slcs in self.slices.items()},
            meta=self.meta,
        )
        section._fs = self._fs
        section._t_from_fs = self._t_from_fs
        # ^ We don't want t to be recalculated from 0, as we want to preserve
        # the original timestamps.

        return section

    def __repr__(self) -> str:
        return f"<Signal of {len(self.y)} values>"

    def copy(
        self,
        *,
        y: _ca.Iterable | None = None,
        unit: str | None = None,
        label: str | None = None,
        chpoints: ChPoints | None = None,
        marks: dict[str, _ca.Iterable[int]] | None = None,
        slices: _SlicesInput | None = None,
        meta: dict[str, _t.Any] | None = None,
    ) -> "Signal":
        """Creates a copy of the signal (all :mod:`numpy` arrays are copied).

        Fields passed as arguments will be replaced.
        """
        return type(self)(
            y=(self.y.copy() if y is None else y),
            unit=self.unit if unit is None else unit,
            t=None if self._t is None else self._t.copy(),
            fs=None if self._t is not None else self._fs,
            label=self.label if label is None else label,
            chpoints=(_cp.deepcopy(self.chpoints) if chpoints is None else chpoints),
            marks=(_cp.deepcopy(self.marks.data) if marks is None else marks),
            slices=(_cp.deepcopy(self._slices.data) if slices is None else slices),
            meta=((self.meta and _cp.deepcopy(self.meta)) if meta is None else meta),
        )

    def shift_t(self, dt: float | None = None) -> "Signal":
        """Shift timestamps with a delta of the same unit as the signal.
        If not provided, timestamps are made to start at zero.
        """
        # Avoid modification of original `t` if this is only a view
        if self.t.base is not None:  # Here t calculation happens
            if self._t is not None:  # This is always true at this point
                self._t = self._t.copy()
        if self._t is not None:  # This is always true at this point
            if dt is not None:
                self._t += dt
            else:
                self._t -= self._t[0]
        return self

    def t2i(self, t_s: float) -> int:
        """Finds the index closest to the given time point.

        Negative ``t_s`` is counted from the end.

        :param t_s: time point in seconds
        :return: index

        .. seealso::
            :meth:`by_t`
        """
        if t_s < 0:
            t_s = self.t[-1] + t_s
        return _np.abs(self.t - t_s).argmin()

    def iter_ccycle_slices(self) -> _ca.Iterator[slice]:
        """Generator for slices of full cardiac cycles.
        Ignores the signal sections before the first and after the last onset.

        Raises a warning, if there are no onsets.
        Empty, if there is only 1 onset.
        """
        if self.onsets.size == 0:
            warnings.warn("Onsets not yet detected.")
        for start, end in zip(self.onsets, self.onsets[1:]):
            yield slice(start, end)

    def iter_ccycles(self) -> _ca.Iterator["Signal"]:
        """Generator for full cardiac cycles.
        Ignores the signal sections before the first and after the last onset.

        .. warning::
            Generates *views*, that is, before modification, consider calling
            :meth:`copy`.

        Raises a warning, if there are no onsets.
        Empty, if there is only 1 onset.
        """
        yield from map(self.__getitem__, self.iter_ccycle_slices())

    @property
    def ccycles(self) -> _CCyclesIndexer:
        """Allows selecting signal parts by the indices of cardiac cycles.

        Ignores the signal sections before the first and after the last onset.

        Examples::

            signal.ccycles[1]
            signal.ccycles[1:]
            signal.ccycles[1:4]
            signal.ccycles[1:10:2]

        :returns: an indexer object capable of slicing:
            Setting signal parts is not supported.
        :raises IndexError: when the index is equal or greater than the
            index of the last onset in :attr:`onsets`.
        """
        return _CCyclesIndexer(self)

    @property
    def by_t(self) -> _ByTIndexer:
        """Allows selecting signal parts by time points.

        ``step`` of slice is not supported. Time points can be non-integer
        or from the end::

            signal.by_t[2.0:-3.5]

        .. note::
            *mypy* gives a false positive error, use ``# type: ignore[misc]``.

        :returns: an indexer object capable of slicing:
            Setting signal parts is not supported.
        """
        return _ByTIndexer(self)

    @property
    def by_onset(self) -> _InclusiveByOnsetIndexer:
        """Allows selecting signal parts by onsets.

        ``step`` of slice is not supported. If ``start`` or ``stop`` is ``None``,
        then they refer to the first and the last onset respectively.

        .. note::
            Selection is inclusive, the specified end onset point is also
            included, to mark that the last part is a full cardiac cycle too.

        :returns: an indexer object capable of slicing:
            Setting signal parts is not supported.
        """
        return _InclusiveByOnsetIndexer(self)

    def plot(
        self,
        ax: _plt.Axes | None = None,
        *fmt_args,
        onsets: bool = True,
        points: bool | set[str] = True,
        marks: bool | set[str] = True,
        legend: str = "auto",
        title: str | None = None,
        t0: float | None = None,
        **plot_kw,
    ) -> _plt.Axes:
        """Plots the signal.

        :param ax: target axes. If ``None``, a new figure is created.
        :param onsets: show onsets.
        :param points: show other points. ``True`` to plot all, ``set`` of names
            to filter (see :class:`CpIndices`).
        :param marks: show marks. ``True`` to plot all, ``set`` of names
            to filter.
        :param legend: ``auto``: show legend when needed,
            ``off``: don't show, ``outside``: move legend out of plot area.
        :param title: plot title.
        :param t0: shift timestamps to start from ``t0``, without copying or
            modifying the object (unlike :meth:`shift_t`).

            .. versionadded:: 0.0.3
        :param fmt_args: line style and color shorthand for :func:`visu.plot_signal`.
        :param plot_kw: keyword arguments for :func:`visu.plot_signal`.
        :return: the axes object.

        .. versionchanged:: 0.0.3
            Grid is on by default.
        """
        if not ax:
            _, ax = _plt.subplots()
        assert ax  # Make mypy happy

        if self.label and "label" not in plot_kw:
            plot_kw = {"label": self.label, **plot_kw}
        has_legend = "label" in plot_kw

        _v.plot_signal(
            ax,
            self.t if t0 is None else self.t - self.t[0] + t0,
            self.y,
            *fmt_args,
            y_unit=self.unit,
            t_unit="s",
            **plot_kw,
        )

        if points and self.chpoints:
            points_: bool | set[str]
            if onsets:
                points_ = set(CpIndices.NAMES)
                points_ = points_ - {"onset"}
            else:
                points_ = points
            self.chpoints.plot(ax, t=self.t, y=self.y, points=points_)
            has_legend = True
        if onsets and self.chpoints and len(self.onsets):
            ax.plot(
                self.t_onsets,
                self.y[self.onsets],
                "r+",
                label=f"onset ({self.chpoints.alg})",
            )
            for i in range(0, len(self.onsets), 5):
                ax.text(
                    self.t_onsets[i],
                    self.y[self.onsets[i]],
                    str(i),
                    verticalalignment="top",
                    fontsize=10,
                )
            has_legend = True
        if marks and self.marks:
            bottom, top = ax.get_ylim()
            for name, indices in self.marks.items():
                if len(indices) and (marks is True or name in marks):
                    fake_line = ax.plot([], [])[0]  # Hack to increment color cycle
                    ax.vlines(
                        self.t[indices],
                        ymin=bottom,
                        ymax=top,
                        colors=fake_line.get_color(),
                        lw=0.5,
                        label=name,
                    )
            has_legend = True
        if legend != "off" and has_legend:
            if legend == "outside":
                ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            else:
                ax.legend()
        if title:
            ax.set(title=title)

        ax.grid(True)

        return ax

    def to_hdf(self, group: h5py.Group) -> None:
        """Saves contents to a HDF5 file or a group thereof.

        :param group: :class:`h5py.File` or :class:`h5py.Group`
        """
        group.attrs["format_version"] = self.HDF_FORMAT_VERSION
        group.attrs["type"] = self._fully_qual_name()
        group.attrs["package_version"] = __version__

        if self.label:
            group.attrs["label"] = self.label

        y_ds = group.create_dataset("y", data=self.y)
        y_ds.attrs["unit"] = self.unit

        if not self.t_from_fs:
            t_ds = group.create_dataset("t", data=self.t)
            t_ds.attrs["unit"] = "s"
        group.attrs["t_from_fs"] = self.t_from_fs
        group.attrs["fs"] = self.fs

        if self.chpoints:
            index_table = _np.array([ci.to_array() for ci in self.chpoints.indices])
            seg_ds = group.create_dataset("chpoints", data=index_table)
            seg_ds.attrs["columns"] = [f.name for f in _dc.fields(CpIndices)]
            seg_ds.attrs["alg"] = self.chpoints.alg
            seg_ds.attrs["version"] = self.chpoints.version
            seg_ds.attrs["params"] = json.dumps(self.chpoints.params)

        if self.marks:
            marks_gr = group.create_group("marks")
            for name, indices in self.marks.items():
                marks_gr.create_dataset(name, data=indices)

        if self.slices:
            slices_gr = group.create_group("slices")
            for name, slices in self.slices.items():
                slices_gr.create_dataset(name, data=[[s.start, s.stop] for s in slices])

        if self.meta:
            meta_gr = group.create_group("meta")
            for name, value in self.meta.items():
                meta_gr.attrs[name] = value

    @classmethod
    def from_hdf(cls, group: h5py.Group) -> "Signal":
        """Creates an instance from a HDF5 file or group.

        :param group: :class:`h5py.File` or :class:`h5py.Group`
        """
        if (v := group.attrs["type"]) != cls._fully_qual_name():
            raise ValueError(f"Type mismatch: file {v} class: {cls._fully_qual_name()}")
        if (v := group.attrs["format_version"]) != cls.HDF_FORMAT_VERSION:
            raise ValueError(
                f"Format version mismatch: "
                f"file: {v} class: {cls.HDF_FORMAT_VERSION}"
            )

        t_from_fs = group.attrs["t_from_fs"]
        y = group["y"]

        if seg_ds := group.get("chpoints"):
            indices = [
                CpIndices(**{c: i for c, i in zip(seg_ds.attrs["columns"], row)})
                for row in seg_ds
            ]
            chpoints = ChPoints(
                indices=indices,
                alg=seg_ds.attrs["alg"],
                version=seg_ds.attrs["version"],
                params=json.loads(seg_ds.attrs["params"]),
            )
        else:
            chpoints = None
        return cls(
            y=y,
            unit=y.attrs["unit"],
            t=None if t_from_fs else group["t"],
            fs=None if not t_from_fs else group.attrs["fs"],
            label=group.attrs.get("label"),
            chpoints=chpoints,
            marks=(
                {name: indices for name, indices in g.items()}
                if (g := group.get("marks"))
                else None
            ),
            slices=(
                {
                    name: [slice(start, stop) for start, stop in slices]
                    for name, slices in g.items()
                }
                if (g := group.get("slices"))
                else None
            ),
            meta=(
                {name: value for name, value in g.attrs.items()}
                if (g := group.get("meta"))
                else None
            ),
        )

    @classmethod
    def _fully_qual_name(cls) -> str:
        return f"{cls.__module__}.{cls.__qualname__}"

    def _validate_timestamps(self) -> None:
        assert self._t is not None
        if len(self._t) != len(self._y):
            raise ValueError("`y` and `t` must have the same length")
        unique, counts = _np.unique(self._t, return_counts=True)
        if len(dup := unique[counts > 1]):
            raise ValueError(f"Duplicate timestamps found: {dup.tolist()}")
        dt = _np.diff(self._t)
        if len(back := _np.nonzero(dt < 0)[0]):
            raise ValueError(
                f"Timestamps are not monotone increasing at indices {back.tolist()}"
            )


def _validate_1d(a: _t.Any, name: str) -> None:
    if (d := _np.ndim(a)) != 1:
        raise ValueError(f"`{name}` must be 1D, got {d}D")
