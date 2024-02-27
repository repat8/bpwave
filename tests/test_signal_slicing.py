import numpy as np
import pytest

from bpwave import Signal


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
