import dataclasses

import numpy as np
import pytest

from bpwave import CpIndices


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


def test__cpindices__names() -> None:
    assert CpIndices.NAMES == tuple(f.name for f in dataclasses.fields(CpIndices))


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
        onset=CpIndices.UNSET,
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
        refl_onset=CpIndices.UNSET,
        refl_peak=CpIndices.UNSET,
        dicr_notch=160 - offset,
        dicr_peak=170 - offset,
    )


def test__cpindices__add__in_range(cpindices_all_set: CpIndices) -> None:
    offset = 10
    assert cpindices_all_set + offset == CpIndices(
        onset=100 + offset,
        sys_peak=120 + offset,
        refl_onset=140 + offset,
        refl_peak=150 + offset,
        dicr_notch=160 + offset,
        dicr_peak=170 + offset,
    )


def test__cpindices__add__unset(cpindices_with_unset: CpIndices) -> None:
    offset = 10
    assert cpindices_with_unset + offset == CpIndices(
        onset=100 + offset,
        sys_peak=120 + offset,
        refl_onset=CpIndices.UNSET,
        refl_peak=CpIndices.UNSET,
        dicr_notch=160 + offset,
        dicr_peak=170 + offset,
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
