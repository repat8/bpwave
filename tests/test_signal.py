import pathlib as pl

import h5py  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import pytest

from bpwave import ChPoints, CpIndices, Signal, visu


def test__signal__t__calc() -> None:
    y = [0, 1, 2, 1, 0, 1, 2, 1, 0]
    s = Signal(y=y, unit="px", fs=30)
    assert s.y.tolist() == y
    assert s.fs == 30.0
    assert s.unit == "px"
    assert np.allclose(s.t, np.arange(len(y), dtype=float) / 30.0)
    assert len(y) == len(s.y) == len(s.t)


def test__signal__fs__calc() -> None:
    y = [0, 1, 2, 1, 0, 1, 2, 1, 0]
    s = Signal(y=y, unit="px", t=np.arange(len(y)) / 30)
    assert s.y.tolist() == y
    assert s.fs == 30.0
    assert s.unit == "px"
    assert len(y) == len(s.y) == len(s.t)


def test__signal__t_fs_validation() -> None:
    with pytest.raises(ValueError):
        Signal(y=[1], t=[0.0], fs=10.0)
    with pytest.raises(ValueError):
        Signal(y=[1], t=[0.0, 1.0])
    with pytest.raises(ValueError):
        Signal(y=[1, 2], t=[0.0, 0.0])
    with pytest.raises(ValueError):
        Signal(y=[1, 2], t=[1.0, 0.0])


def test__signal__y__setter(simple_signal: Signal) -> None:
    new_values = simple_signal.y.copy() + 2
    simple_signal.y = new_values.copy()
    assert (simple_signal.y == new_values).all()
    simple_signal.y -= 1
    assert (simple_signal.y == new_values - 1).all()
    with pytest.raises(ValueError):
        simple_signal.y = new_values[1:]


def test__signal__y__mypy(simple_signal: Signal) -> None:
    simple_signal.y = np.array(simple_signal.y)
    simple_signal.y = simple_signal.y.tolist()
    simple_signal.copy(y=np.array(simple_signal.y))
    simple_signal.copy(y=simple_signal.y.tolist())


def test__signal__val_1d(simple_signal: Signal) -> None:
    scalar = np.median([[1, 2], [1, 2]])  # Intentionally no axis
    with pytest.raises(ValueError):
        Signal(y=scalar, fs=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Signal(y=scalar, t=scalar)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        simple_signal.y = 1  # type: ignore[assignment]
    with pytest.raises(ValueError):
        simple_signal.copy(y=scalar)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        simple_signal.marks = {"test": scalar}  # type: ignore[assignment]


def test__signal__features__validation() -> None:
    y = [0, 1, 2, 1, 0, 1, 2, 1, 0]
    s = Signal(
        y=y,
        fs=30,
        chpoints=ChPoints(
            indices=[CpIndices(onset=0)], alg="x", version="x", params={}
        ),
    )
    assert s.onsets.tolist() == [0]
    s.chpoints = ChPoints(indices=[CpIndices(onset=4)], alg="x", version="x", params={})
    assert s.onsets.tolist() == [4]
    with pytest.raises(ValueError):
        Signal(
            y=y,
            fs=30,
            chpoints=ChPoints(
                indices=[CpIndices(onset=100)], alg="x", version="x", params={}
            ),
        )
    with pytest.raises(ValueError):
        s.chpoints = ChPoints(
            indices=[CpIndices(onset=100)], alg="x", version="x", params={}
        )


def test__signal__onsets__empty() -> None:
    assert (Signal(y=[1, 2], fs=20).onsets == np.array([], int)).all()
    assert Signal(y=[1, 2], fs=20).onsets.size == 0


def test__signal__marks(simple_signal: Signal) -> None:
    simple_signal.marks["x"] = [1, 2]
    assert np.all(simple_signal.marks["x"] == np.array([1, 2]))
    with pytest.raises(ValueError):
        simple_signal.marks = {"b": [100]}  # type: ignore
    with pytest.raises(ValueError):
        simple_signal.marks["b"] = [100]


def test__signal__slices(simple_signal: Signal) -> None:
    n = len(simple_signal.y)
    simple_signal.slices["x"] = [np.s_[1 : n - 1]]
    assert simple_signal.slices["x"] == [np.s_[1 : n - 1]]
    simple_signal.slices = {"y": [np.s_[2:n]]}  # type: ignore[assignment]
    assert simple_signal.slices["y"] == [np.s_[2:n]]
    with pytest.raises(ValueError):
        simple_signal.slices = {"b": [np.s_[0 : n + 1]]}  # type: ignore[assignment]
    with pytest.raises(ValueError):
        simple_signal.slices["b"] = [np.s_[-10:n]]


def test__signal__copy(simple_signal: Signal) -> None:
    s2 = simple_signal.copy()
    assert s2.y.base is None
    assert (s2.y == simple_signal.y).all()
    assert s2.y is not simple_signal.y
    assert s2.t.base is None
    assert np.allclose(s2.t, simple_signal.t)
    assert s2.chpoints is not None
    assert s2.chpoints == simple_signal.chpoints
    assert s2.chpoints is not simple_signal.chpoints
    assert s2.chpoints.indices[0] is not simple_signal.chpoints.indices[0]
    assert s2.marks == simple_signal.marks
    assert s2.marks is not simple_signal.marks
    assert s2.slices == simple_signal.slices
    assert s2.slices is not simple_signal.slices
    assert s2.meta == simple_signal.meta
    assert s2.meta is not simple_signal.meta


def test__signal__copy__with_replace(simple_signal: Signal) -> None:
    y2 = [0, 1, -3, 1, 0, 1, -3, 1, 0]
    s2 = simple_signal.copy(y=y2)
    assert s2.y.base is None
    assert (s2.y == y2).all()
    assert s2.t.base is None
    assert np.allclose(s2.t, simple_signal.t)
    assert s2.chpoints is not None
    assert s2.chpoints == simple_signal.chpoints
    assert s2.chpoints is not simple_signal.chpoints
    assert s2.chpoints.indices[0] is not simple_signal.chpoints.indices[0]
    with pytest.raises(ValueError):
        simple_signal.copy(y=[0])


def test__signal__shift_t(simple_signal: Signal) -> None:
    orig = simple_signal.copy()
    part = simple_signal[2:].copy()
    assert part.shift_t().t[0] == 0.0
    part = simple_signal[2:]
    assert part.shift_t().t[0] == 0.0
    assert np.allclose(simple_signal.t, orig.t)
    assert np.allclose(simple_signal.shift_t(10.0).t, orig.t + 10.0)


def test__signal__t2i(simple_signal: Signal) -> None:
    y = [1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 3]
    t = np.array(np.arange(len(y)), float) / 30.0
    s = Signal(y=y, unit="px", t=t)
    i = s.t2i(1 / 30.0 * 5)
    assert isinstance(i, (int, np.integer))
    assert i == 5


@pytest.mark.human
def test__signal__plot(simple_signal: Signal) -> None:
    with visu.figure(nrows=4, title="test__signal__plot", figsize=(5, 10)) as (
        _,
        axes,
    ):
        axes = axes.ravel()
        simple_signal.plot(axes[0], title="all on")
        simple_signal.plot(axes[1], title="onsets off", onsets=False)
        simple_signal.plot(axes[2], title="points off", points=False)
        simple_signal.plot(
            axes[3],
            title="filtered points, t0=10",
            points={"sys_peak"},
            onsets=False,
            t0=10,
        )
    plt.show()


def test__signal__hdf__t_from_fs_has_segmentation(
    simple_signal: Signal,
    out_folder: pl.Path,
) -> None:
    path = out_folder / "signal1.hdf5"
    with h5py.File(path, "w") as out_file:
        simple_signal.to_hdf(out_file)
    with h5py.File(path, "r") as in_file:
        s_loaded = Signal.from_hdf(in_file)
    assert np.allclose(s_loaded.y, simple_signal.y)
    assert s_loaded.fs == simple_signal.fs
    assert s_loaded.label == simple_signal.label
    assert s_loaded.unit == simple_signal.unit
    assert s_loaded.t_from_fs
    assert s_loaded.chpoints == simple_signal.chpoints
    assert s_loaded.marks == simple_signal.marks
    assert s_loaded.slices == simple_signal.slices
    assert s_loaded.meta == simple_signal.meta


def test__signal__repr(simple_signal: Signal) -> None:
    assert repr(simple_signal) == "<Signal of 9 values>"


def test__signal__str(simple_signal: Signal) -> None:
    assert str(simple_signal) == "<Signal of 9 values>"
