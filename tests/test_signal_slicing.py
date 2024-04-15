import numpy as np
import pytest

from bpwave import ChPoints, CpIndices, Signal


def test__signal__getitem(simple_signal: Signal) -> None:
    s2 = simple_signal[1:-3]
    assert s2.y.base is simple_signal.y
    assert s2.y.tolist() == [1, 2, 1, 0, 1]
    assert s2.t.base is simple_signal.t
    assert np.allclose(s2.t, np.array([1, 2, 3, 4, 5], float) / 30.0)
    assert s2.onsets.tolist() == [3]
    assert s2.chpoints == ChPoints(
        alg="manual",
        params={},
        version="0",
        indices=[
            CpIndices(onset=0, sys_peak=2) - 1,
            CpIndices(onset=4, sys_peak=-1) - 1,
        ],
    )
    assert s2.marks == {"a": [0]}
    assert s2.slices == {"a": [slice(1, 3)]}
    assert s2.meta == simple_signal.meta


def test__signal__getitem__validation(simple_signal: Signal) -> None:
    with pytest.raises(NotImplementedError):
        simple_signal[0]  # type: ignore # noqa
    with pytest.raises(NotImplementedError):
        simple_signal[0:2:2]  # noqa


def test__signal__getitem__start(simple_signal: Signal) -> None:
    s2 = simple_signal[-6:]
    assert s2.y.tolist() == [1, 0, 1, 2, 1, 0]
    assert s2.chpoints == ChPoints(
        alg="manual",
        params={},
        version="0",
        indices=[
            CpIndices(onset=4, sys_peak=6) - (len(simple_signal.y) - 6),
            CpIndices(onset=8) - (len(simple_signal.y) - 6),
        ],
    )
    assert s2.marks.keys() == {"a"}
    assert s2.marks["a"].size == 0


def test__signal__getitem__stop(simple_signal: Signal) -> None:
    s2 = simple_signal[:4]
    assert s2.y.tolist() == [0, 1, 2, 1]
    assert s2.chpoints == ChPoints(
        alg="manual",
        params={},
        version="0",
        indices=[
            CpIndices(onset=0, sys_peak=2),
        ],
    )


def test__signal__getitem__empty_slice_removed() -> None:
    s = Signal(
        y=np.ones(30),
        fs=30,
        slices={"test": [slice(5, 10), slice(10, 15), slice(15, 20)]},
    )
    s2 = s[10:15]
    assert s2.slices["test"] == [slice(0, 5)]
    s2 = s[9:16]
    assert s2.slices["test"] == [slice(0, 1), slice(1, 6), slice(6, 7)]


def test__signal__by_t__start_stop(bp_signal: Signal) -> None:
    section = bp_signal.by_t[2.0:3.5]  # type: ignore[misc]
    assert np.isclose(section.t[0], 2.0, atol=1 / section.fs)
    assert np.isclose(section.t[-1], 3.5, atol=1 / section.fs)


def test__signal__by_t__start(bp_signal: Signal) -> None:
    section = bp_signal.by_t[2.0:]  # type: ignore[misc]
    assert np.isclose(section.t[0], 2.0, atol=1 / section.fs)
    assert section.t[-1] == bp_signal.t[-1]


def test__signal__by_t__stop(bp_signal: Signal) -> None:
    section = bp_signal.by_t[:3.5]  # type: ignore[misc]
    assert np.isclose(section.t[-1], 3.5, atol=1 / section.fs)
    assert section.t[0] == bp_signal.t[0]


def test__signal__by_t__all(bp_signal: Signal) -> None:
    section = bp_signal.by_t[:]
    assert (section.t == bp_signal.t).all()


def test__signal__by_t__back(bp_signal: Signal) -> None:
    section = bp_signal.by_t[-5:-1]  # type: ignore[misc]
    assert np.isclose(section.t[0], bp_signal.t[-1] - 5, atol=1 / section.fs)
    assert np.isclose(section.t[-1], bp_signal.t[-1] - 1, atol=1 / section.fs)


def test__signal__by_t__validation(bp_signal: Signal) -> None:
    with pytest.raises(ValueError):
        bp_signal.by_t[::1]  # noqa


def test__signal__by_onset__start_stop(bp_signal: Signal) -> None:
    section = bp_signal.by_onset[2:5]
    assert (
        section.y == bp_signal.y[bp_signal.onsets[2] : bp_signal.onsets[5] + 1]
    ).all()


def test__signal__by_onset__start(bp_signal: Signal) -> None:
    section = bp_signal.by_onset[2:]
    assert (
        section.y == bp_signal.y[bp_signal.onsets[2] : bp_signal.onsets[-1] + 1]
    ).all()


def test__signal__by_onset__stop(bp_signal: Signal) -> None:
    section = bp_signal.by_onset[:5]
    assert (
        section.y == bp_signal.y[bp_signal.onsets[0] : bp_signal.onsets[5] + 1]
    ).all()


def test__signal__by_onset__all(bp_signal: Signal) -> None:
    section = bp_signal.by_onset[:]
    assert (
        section.y == bp_signal.y[bp_signal.onsets[0] : bp_signal.onsets[-1] + 1]
    ).all()


def test__signal__by_onset__back(bp_signal: Signal) -> None:
    section = bp_signal.by_onset[-3:-1]
    assert (
        section.y == bp_signal.y[bp_signal.onsets[-3] : bp_signal.onsets[-1] + 1]
    ).all()


def test__signal__by_onset__validation(bp_signal: Signal) -> None:
    with pytest.raises(ValueError):
        bp_signal.by_onset[::1]  # noqa


def test__signal__iter_ccycle_slices(simple_signal: Signal) -> None:
    with pytest.warns():
        s = Signal(y=[1, 2], fs=10.0)
        assert s.onsets.size == 0
        assert not list(s.iter_ccycle_slices())
    assert list(simple_signal.iter_ccycle_slices()) == [
        slice(0, 4),
        slice(4, 8),
    ]


def test__signal__iter_periods(simple_signal: Signal) -> None:
    assert [p.y.tolist() for p in simple_signal.iter_ccycles()] == [
        [0.0, 1.0, 2.0, 1.0],
        [0.0, 1.0, 2.0, 1.0],
    ]


def test__signal__ccycles(simple_signal: Signal) -> None:
    c1 = simple_signal.ccycles[1]
    assert c1.chpoints is not None
    assert len(c1.chpoints.indices) == 1
    assert np.allclose(
        c1.y, simple_signal.y[simple_signal.onsets[1] : simple_signal.onsets[2]]
    )
    c2 = simple_signal.ccycles[:2]
    assert len(c2) == 2
    for c in c2:
        assert c.chpoints is not None
        assert len(c.chpoints.indices) == 1
