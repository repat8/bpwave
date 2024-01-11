import h5py  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import pytest

from bpwave import ChPoints, CpIndices, Signal, visu

from . import utils


@pytest.fixture(scope="function")
def cpindices_all_set() -> CpIndices:
    return CpIndices(
        onset=100,
        sys_peak=120,
        refl_onset=140,
        refl_peak=150,
        dicr_notch=160,
        dicr_peak=170,
    )


@pytest.fixture(scope="function")
def cpindices_with_unset() -> CpIndices:
    return CpIndices(
        onset=100,
        sys_peak=120,
        dicr_notch=160,
        dicr_peak=170,
    )


@pytest.fixture(scope="function")
def simple_signal() -> Signal:
    return Signal(
        y=[0, 1, 2, 1, 0, 1, 2, 1, 0],
        fs=30.0,
        chpoints=ChPoints(
            alg="manual",
            params={},
            version="0",
            indices=[
                CpIndices(onset=0, sys_peak=2),
                CpIndices(onset=4, sys_peak=6),
                CpIndices(onset=8),
            ],
        ),
        label="Simple signal",
        marks={"a": [1]},
        meta={"source": "test"},
    )


def test__cpindices__empty() -> None:
    with pytest.warns():
        CpIndices()


def test__cpindices__min(cpindices_with_unset) -> None:
    assert cpindices_with_unset.min() == 100
    assert cpindices_with_unset.max() == 170


def test__cpindices__sub__in_range(cpindices_all_set: CpIndices) -> None:
    offset = 10
    assert cpindices_all_set - offset == CpIndices(
        onset=100 - offset,
        sys_peak=120 - offset,
        refl_onset=140 - offset,
        refl_peak=150 - offset,
        dicr_notch=160 - offset,
        dicr_peak=170 - offset,
    )


def test__cpindices__sub__out_of_range(cpindices_all_set: CpIndices) -> None:
    offset = 120
    assert cpindices_all_set - offset == CpIndices(
        onset=-1,
        sys_peak=0,
        refl_onset=140 - offset,
        refl_peak=150 - offset,
        dicr_notch=160 - offset,
        dicr_peak=170 - offset,
    )


def test__cpindices__sub__unset(cpindices_with_unset: CpIndices) -> None:
    offset = 10
    assert cpindices_with_unset - offset == CpIndices(
        onset=100 - offset,
        sys_peak=120 - offset,
        refl_onset=-1,
        refl_peak=-1,
        dicr_notch=160 - offset,
        dicr_peak=170 - offset,
    )


def test__cpindices__without_unset(cpindices_with_unset: CpIndices) -> None:
    assert cpindices_with_unset.without_unset() == {
        "onset": 100,
        "sys_peak": 120,
        "dicr_notch": 160,
        "dicr_peak": 170,
    }


def test__cpindices__to_array(cpindices_with_unset: CpIndices) -> None:
    assert (
        cpindices_with_unset.to_array() == np.array([100, 120, -1, -1, 160, 170])
    ).all()


def test__cpindices__clamped(cpindices_with_unset: CpIndices) -> None:
    assert cpindices_with_unset.clamped() == cpindices_with_unset
    assert (
        cpindices_with_unset.clamped(start=cpindices_with_unset.onset)
        == cpindices_with_unset
    )
    assert cpindices_with_unset.clamped(start=110) == CpIndices(
        onset=-1,
        sys_peak=120,
        dicr_notch=160,
        dicr_peak=170,
    )
    assert cpindices_with_unset.clamped(start=110, stop=170) == CpIndices(
        onset=-1,
        sys_peak=120,
        dicr_notch=160,
        dicr_peak=-1,
    )


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


@pytest.mark.human
def test__signal__plot(simple_signal: Signal) -> None:
    with visu.figure(nrows=4, title="test__signal__plot", figsize=(5, 10)) as (
        _,
        axes,
    ):
        simple_signal.plot(axes[0], title="all on")
        simple_signal.plot(axes[1], title="onsets off", onsets=False)
        simple_signal.plot(axes[2], title="points off", points=False)
        simple_signal.plot(
            axes[3], title="filtered points", points={"sys_peak"}, onsets=False
        )
    plt.show()


def test__signal__hdf__t_from_fs_has_segmentation(simple_signal):
    path = utils.OUT_FOLDER / "signal1.hdf5"
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
    assert s_loaded.meta == simple_signal.meta
