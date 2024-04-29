import matplotlib.pyplot as plt
import pytest

from bpwave import CpIndices, Signal


def test__chpoints__getitem_val(bp_signal: Signal) -> None:
    with pytest.raises(NotImplementedError):
        bp_signal.chpoints[0]  # noqa
    with pytest.raises(NotImplementedError):
        bp_signal.chpoints[0:100:2]  # noqa
    with pytest.raises(NotImplementedError):
        bp_signal.chpoints[-100:-1]  # noqa


def test__chpoints__getitem__start_stop(bp_signal: Signal) -> None:
    cp = bp_signal.chpoints
    part = cp[cp.indices[1].sys_peak : cp.indices[-1].sys_peak]
    assert len(part.indices) == len(bp_signal.chpoints.indices) - 1
    assert part.indices[0].onset == CpIndices.UNSET
    assert part.indices[0].sys_peak != CpIndices.UNSET
    assert part.indices[-1].onset != CpIndices.UNSET
    assert part.indices[-1].sys_peak == CpIndices.UNSET


def test__chpoints__getitem__start(bp_signal: Signal) -> None:
    cp = bp_signal.chpoints
    part = cp[cp.indices[1].sys_peak :]
    assert len(part.indices) == len(bp_signal.chpoints.indices) - 1
    assert part.indices[0].onset == CpIndices.UNSET
    assert part.indices[0].sys_peak != CpIndices.UNSET
    assert part.indices[-1].onset != CpIndices.UNSET
    assert part.indices[-1].sys_peak != CpIndices.UNSET


def test__chpoints__getitem__stop(bp_signal: Signal) -> None:
    cp = bp_signal.chpoints
    part = cp[: cp.indices[-1].sys_peak]
    assert len(part.indices) == len(bp_signal.chpoints.indices)
    assert part.indices[0].onset != CpIndices.UNSET
    assert part.indices[0].sys_peak != CpIndices.UNSET
    assert part.indices[-1].onset != CpIndices.UNSET
    assert part.indices[-1].sys_peak == CpIndices.UNSET


@pytest.mark.human
def test__chpoints__plot(bp_signal: Signal) -> None:
    cp = bp_signal.chpoints[500:5000]
    sp = bp_signal[500:5000]
    ax = sp.plot(points=False, onsets=False, title="test__chpoints__plot")
    cp.plot(ax, t=sp.t, y=sp.y)
    plt.show()
