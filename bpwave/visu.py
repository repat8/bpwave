"""Visualization utils."""
import contextlib
import typing as _t
from collections.abc import Generator

import matplotlib.pyplot as _plt
import numpy as _np

if _t.TYPE_CHECKING:
    from .signal import Signal


def plot_signal(
    ax: _plt.Axes,
    t: _np.ndarray,
    y: _np.ndarray,
    *fmt_args,
    t_unit: str = "s",
    y_unit: str = "any",
    append: bool = False,
    **plot_kw,
) -> None:
    """Plots a digital signal.

    :param ax: the target axis object
    :param t: time values
    :param y: signal values
    :param fmt_args: line format arg
    :param t_unit: unit of time axis
    :param y_unit: unit of signal values
    :param append: plot the line only, other settings were done in a previous call
    :param plot_kw: any other args, like ``linewidth``, etc.
    """
    ax.step(t, y, *fmt_args, where="mid", linewidth=0.5, **plot_kw)
    if not append:
        ax.set(
            xlabel=f"$t$ [{t_unit}]",
            ylabel=f"$y$ [{y_unit}]",
            xlim=(t.min(), t.max()),
        )


def plot_signal_slices(
    signal: "Signal", /, **subplots_kw
) -> tuple[_plt.Axes, _plt.Axes]:
    """Visualizes :attr:`bpwave.Signal.slices`.

    :param signal: the signal object with :attr:`slices` filled.
    :param subplots_kw: keyword args to be passed for the two subplots.
    :return: (signal subplot, slices subplot)
    """
    fig, (ax_sig, ax_slc) = _plt.subplots(
        nrows=2,
        sharex=True,
        **(dict(figsize=(15, 6), gridspec_kw={"height_ratios": [2, 1]}) | subplots_kw),
    )
    signal.plot(ax=ax_sig, legend="off")
    xlabel = ax_sig.get_xlabel()
    ax_sig.set(xlabel=None)
    for i, (name, slices) in enumerate(signal.slices.items()):
        for slc in slices:
            ax_slc.fill_between(
                [signal.t[slc.start], signal.t[slc.stop - 1]],
                [i - 0.4, i - 0.4],
                [i + 0.4, i + 0.4],
                alpha=0.5,
            )
    ax_slc.set(
        yticks=range(len(signal.slices)),
        yticklabels=signal.slices.keys(),
        xlabel=xlabel,
    )
    ax_slc.grid(True)
    fig.tight_layout()
    return ax_sig, ax_slc


@contextlib.contextmanager
def figure(
    nrows: int = 1,
    ncols: int = 1,
    *,
    autogrid: int | None = None,
    title: str | None = None,
    block: bool = False,
    tight_layout: bool = True,
    **subplot_args,
) -> Generator[tuple[_plt.Figure, _np.ndarray], None, None]:
    """
    .. versionchanged:: 0.3.0
        New parameter ``autogrid``.
    """
    if autogrid is not None:
        nrows = max(1, int(_np.floor(_np.sqrt(autogrid))))
        ncols = max(1, int(_np.ceil(_np.sqrt(autogrid))))
    fig, axes = _plt.subplots(nrows=nrows, ncols=ncols, **subplot_args)
    axes = _np.atleast_1d(axes)
    if autogrid is not None and (n_empty := (nrows * ncols - autogrid)):
        for ax in axes.ravel()[-n_empty:]:
            ax.axis("off")
    if title:
        fig.suptitle(title)
    yield fig, axes
    if tight_layout:
        fig.tight_layout()
    _plt.show(block=block)
