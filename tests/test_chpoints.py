import matplotlib.pyplot as plt
import pytest

import bpwave.visu
from bpwave import CpIndices, Signal


def test__chpoints__add(bp_signal: Signal) -> None:
    assert bp_signal.chpoints is not None  # mypy
    offset = 15
    cp = bp_signal.chpoints + offset
    assert cp.indices == [ci + offset for ci in bp_signal.chpoints.indices]
    assert cp.alg == bp_signal.chpoints.alg


def test__chpoints__sub(bp_signal: Signal) -> None:
    assert bp_signal.chpoints is not None  # mypy
    offset = 15
    cp = bp_signal.chpoints - offset
    assert cp.indices == [ci - offset for ci in bp_signal.chpoints.indices]
    assert cp.alg == bp_signal.chpoints.alg


def test__chpoints__getitem_val(bp_signal: Signal) -> None:
    assert bp_signal.chpoints is not None  # mypy
    with pytest.raises(NotImplementedError):
        bp_signal.chpoints[0]  # type: ignore  # noqa
    with pytest.raises(NotImplementedError):
        bp_signal.chpoints[0:100:2]  # noqa
    with pytest.raises(NotImplementedError):
        bp_signal.chpoints[-100:-1]  # noqa


def test__chpoints__getitem__start_stop(bp_signal: Signal) -> None:
    assert bp_signal.chpoints is not None  # mypy
    cp = bp_signal.chpoints
    part = cp[cp.indices[1].sys_peak : cp.indices[-1].sys_peak]
    assert len(part.indices) == len(bp_signal.chpoints.indices) - 1
    assert part.indices[0].onset == CpIndices.UNSET
    assert part.indices[0].sys_peak != CpIndices.UNSET
    assert part.indices[-1].onset != CpIndices.UNSET
    assert part.indices[-1].sys_peak == CpIndices.UNSET


def test__chpoints__getitem__start(bp_signal: Signal) -> None:
    assert bp_signal.chpoints is not None  # mypy
    cp = bp_signal.chpoints
    part = cp[cp.indices[1].sys_peak :]
    assert len(part.indices) == len(bp_signal.chpoints.indices) - 1
    assert part.indices[0].onset == CpIndices.UNSET
    assert part.indices[0].sys_peak != CpIndices.UNSET
    assert part.indices[-1].onset != CpIndices.UNSET
    assert part.indices[-1].sys_peak != CpIndices.UNSET


def test__chpoints__getitem__stop(bp_signal: Signal) -> None:
    assert bp_signal.chpoints is not None  # mypy
    cp = bp_signal.chpoints
    part = cp[: cp.indices[-1].sys_peak]
    assert len(part.indices) == len(bp_signal.chpoints.indices)
    assert part.indices[0].onset != CpIndices.UNSET
    assert part.indices[0].sys_peak != CpIndices.UNSET
    assert part.indices[-1].onset != CpIndices.UNSET
    assert part.indices[-1].sys_peak == CpIndices.UNSET


@pytest.mark.human
def test__chpoints__plot(bp_signal: Signal) -> None:
    assert bp_signal.chpoints is not None  # mypy
    cp = bp_signal.chpoints[500:5000]
    sp = bp_signal[500:5000]
    with bpwave.visu.figure(nrows=2, title="test__chpoints__plot") as (_, axes):
        sp.plot(
            ax=axes[0],
            points=False,
            onsets=False,
            title="points=True",
            legend="outside",
        )
        cp.plot(axes[0], t=sp.t, y=sp.y)
        axes[0].legend(loc="center left", bbox_to_anchor=(1, 0.5))
        sp.plot(
            ax=axes[1],
            points=False,
            onsets=False,
            title="points filtered",
            legend="outside",
        )
        cp.plot(axes[1], t=sp.t, y=sp.y, points={"sys_peak"})
    plt.show()
