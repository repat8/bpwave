import abc
import collections.abc as _ca
import dataclasses as _dc
import datetime as _dt
import itertools
import json
import pathlib as _pl
import typing as _t
import warnings

import numpy as _np

from . import signal as _s


class AlgMeta:
    """Algorithm metadata for documentation purposes in HDF5 files."""

    @property
    def name(self) -> str:
        """Qualified name of the algorithm."""
        return f"{self.__module__}.{type(self).__name__}"

    @property
    def params(self) -> dict[str, _t.Any]:
        """Parameter dict."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


class SignalReader(abc.ABC, AlgMeta):
    """Base class for file to :class:`Signal` converters.

    Instances are callable objects.

    Subclasses may take additional parameters needed for the specific file
    format, e. g.::

        c = FakeCsvConverter(y_column='Pressure', t_column='Timestamp')
        signal, _ = c('read/from/here.csv')

    Implementations should override :meth:`_read`.

    .. versionadded:: 0.0.3
    """

    def __call__(
        self,
        in_path: _pl.Path | str,
        *,
        rel_to: _pl.Path | str | None = None,
    ) -> tuple[_s.Signal, _t.Any]:
        """Reads file at ``in_path`` and constructs a :class:`Signal` from it.

        :param in_path: path of the input file
        :param rel_to: path part not to be included in metadata
        :return: the signal object and implementation-specific other data
        """
        conv_in_path = _pl.Path(in_path)
        signal, other = self._read(conv_in_path)
        signal.meta["source_file"] = str(
            conv_in_path if not rel_to else conv_in_path.relative_to(rel_to)
        )
        signal.meta["source_file_date"] = _dt.datetime.now().isoformat()
        signal.meta["source_file_reader"] = self.name
        signal.meta["source_file_params"] = json.dumps(
            self.params, default=lambda o: o.__class__.__name__
        )
        return signal, other

    @abc.abstractmethod
    def _read(self, in_path: _pl.Path) -> tuple[_s.Signal, _t.Any]:
        """Performs the conversion."""


@_dc.dataclass(kw_only=True)
class CsvReader(SignalReader):
    """Reader intended for simple value list or timestamp - value pair CSV files.

    This implementation is based on :func:`numpy.loadtxt`.
    For more complicated CSV contents, we recommend implementing a loader
    based on :func:`pandas.read_csv`.

    :meth:`__call__` returns the loaded signal and a list of comments found at
    the head of the file, the comment sign will be stripped.

    .. versionadded:: 0.0.3
    """

    t_column: int | str | None
    """Index  (or name in the header) of the column of timestamp values.
    ``None`` indicates no timestamp column, in this case, provide ``fs``.
    """

    y_column: int | str
    """Index (or name in the header) of the column of signal values."""

    fs: float | None = None
    """Sampling frequency if ``t_column is None``."""

    delimiter: str = ","
    """Cell delimiter."""

    has_header: bool = False
    """Whether first line is header."""

    skiprows: int = 0
    """Skip top non-comment lines (over the header, if the file has)."""

    comment: str | None = "#"
    """Comment mark at line start."""

    quotechar: str | None = None
    """Quote character."""

    t_converter: str | _t.Callable[[str], float] = "float"
    """:meth:`datetime.datetime.strptime` time format (e. g. ``"%H:%M:%S.%f"``,
    ``"float"`` or a callable converting string cell value to seconds.
    """

    unit: str = "y"
    """Unit of signal values."""

    def __post_init__(self):
        if self.t_column is None:
            if self.fs is None:
                raise ValueError("`fs` must be specified if there is no `t_column`.")
        else:
            if self.fs is not None:
                warnings.warn("`fs` is ignored, timestamps are given.")
        if (
            isinstance(self.y_column, str) or isinstance(self.t_column, str)
        ) and not self.has_header:
            raise ValueError(
                "Columns can be specified by name if `has_header` is true."
            )

    def _read(self, in_path: _pl.Path) -> tuple[_s.Signal, list[str]]:
        i = 0
        with in_path.open() as file:
            # Collect comments from the beginning of the file.
            # After the first non-comment line, the rest will be just ignored.
            comments = []
            # -1: set later
            y_col_idx = -1 if isinstance(self.y_column, str) else self.y_column
            t_col_idx = -1 if isinstance(self.t_column, str) else self.t_column

            for i, line in enumerate(file):
                if self.comment and line.startswith(self.comment):
                    comments.append(line[1:].strip())
                else:
                    if self.has_header:
                        header_cells = line.strip().split(self.delimiter)
                        if isinstance(self.y_column, str):
                            y_col_idx = header_cells.index(self.y_column)
                        if isinstance(self.t_column, str):
                            t_col_idx = header_cells.index(self.t_column)
                    break

        # Continue reading the unconsumed lines
        assert y_col_idx is not None and y_col_idx >= 0
        with in_path.open() as file:
            lines = itertools.islice(file, i + self.has_header, None)
            if self.t_column is None:
                signal = self._read_non_timestamped(y_col_idx, lines)
            else:
                assert t_col_idx is not None and t_col_idx >= 0
                signal = self._read_timestamped(t_col_idx, y_col_idx, lines)

        signal.label = in_path.stem
        return signal, comments

    def _read_non_timestamped(self, y_col: int, lines: _ca.Iterable[str]) -> _s.Signal:
        y = _np.loadtxt(
            lines,
            delimiter=self.delimiter,
            quotechar=self.quotechar,
            comments=self.comment,
            usecols=y_col,
            ndmin=1,
        )

        signal = _s.Signal(
            fs=self.fs,
            y=y,
            unit=self.unit,
        )

        return signal

    def _date_to_seconds(self, x: str) -> float:
        return _datetime_to_seconds(
            _dt.datetime.strptime(x, self.t_converter),  # type: ignore[arg-type]
        )

    def _read_timestamped(
        self, t_col: int, y_col: int, lines: _ca.Iterable[str]
    ) -> _s.Signal:
        converter: _t.Callable[[str], float] | None
        match self.t_converter:
            case "float":
                converter = None  # Conversion to float is the default
            case str():
                converter = self._date_to_seconds
            case _ if callable(self.t_converter):
                converter = self.t_converter
            case _:
                raise ValueError(f"Invalid `t_converter`: {self.t_converter}")
        converters = None if converter is None else {t_col: converter}

        t, y = _np.loadtxt(
            lines,
            delimiter=self.delimiter,
            quotechar=self.quotechar,
            comments=self.comment,
            encoding="utf8",
            converters=converters,
            usecols=(t_col, y_col),
            ndmin=2,
            unpack=True,
        )  # type: ignore[call-overload]

        signal = _s.Signal(
            t=t,
            y=y,
            unit=self.unit,
        )

        return signal


def _datetime_to_seconds(t: _dt.datetime) -> float:
    return t.microsecond / 1e6 + t.second + t.minute * 60 + t.hour * 3600


def to_csv(
    file_path: _pl.Path | str,
    signal: _s.Signal,
    **savetxt_kw,
) -> set[_pl.Path]:
    """Dumps components of the signal into multiple CSV files.

    The intended use case is making the data available for other technologies
    not easily coping with the HDF5 format.

    .. note::
        For the purpose of serializing a :class:`bpwave.Signal` object for
        later reloading, we recommend :meth:`bpwave.Signal.to_hdf` as it
        preserves all metadata.

    :param file_path: file path of the target CSV file that will contain the
        timestamps and data points. Other CSV files may be created as well,
        to the same folder with multiple extensions indicating the fields of
        ``signal``, in the format ``<original_filename>.<field>[.<key>].csv``.
    :param signal: the signal object to be dumped.
    :param savetxt_kw: arguments for the underlying :func:`numpy.savetxt`.
    :return: a set of paths of the files created.

    .. versionadded:: 0.0.3
    """
    path = _pl.Path(file_path)
    created_paths = {path}
    nl = savetxt_kw.get("newline", "\n")
    delimiter = savetxt_kw.get("delimiter", " ")
    header = nl.join(
        line
        for line in [
            f"unit={signal.unit}",
            f"fs={signal.fs}",
            f"t_from_fs={signal.t_from_fs}",
            f"label={signal.label}",
            savetxt_kw.get("header"),
            f"t{delimiter}y",
        ]
        if line
    )

    _np.savetxt(path, _np.c_[signal.t, signal.y], **{**savetxt_kw, "header": header})
    if signal.chpoints:
        _np.savetxt(
            p := path.with_suffix(".chpoints.csv"),
            _np.array([ci.to_array() for ci in signal.chpoints.indices]),
            fmt="%i",
            **{**savetxt_kw, "header": delimiter.join(_s.CpIndices.NAMES)},
        )
        created_paths.add(p)
    if signal.marks:
        for name, indices in signal.marks.items():
            _np.savetxt(
                p := path.with_suffix(f".marks.{name}.csv"),
                indices,
                fmt="%i",
                **{**savetxt_kw, "header": "i"},
            )
            created_paths.add(p)
    if signal.slices:
        for name, slices in signal.slices.items():
            _np.savetxt(
                p := path.with_suffix(f".slices.{name}.csv"),
                [[s.start, s.stop] for s in slices],
                fmt="%i",
                **{**savetxt_kw, "header": f"start{delimiter}stop"},
            )
            created_paths.add(p)
    if signal.meta:
        _np.savetxt(
            p := path.with_suffix(".meta.csv"),
            [
                [k, str(json.JSONEncoder(default=str).encode(v))]
                for k, v in signal.meta.items()
            ],
            fmt="%s",
            **{**savetxt_kw, "header": f"key{delimiter}value"},
        )

        created_paths.add(p)
    return created_paths
