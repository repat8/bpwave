import pathlib as pl

import pytest

from bpwave import CsvReader, Signal, to_csv


def test__csv_reader__timestamped_float_t(data_folder: pl.Path) -> None:
    reader = CsvReader(
        t_column=1,
        y_column=0,
        delimiter=";",
        t_converter=lambda t: float(t) / 1000,
        has_header=True,
    )
    s, comments = reader(data_folder / "timestamped_float_comments.csv")
    _test_timestamped_float_comments_content(s, comments)


def test__csv_reader__timestamped_float_t_by_name(data_folder: pl.Path) -> None:
    reader = CsvReader(
        t_column="t_ms",
        y_column="p",
        delimiter=";",
        t_converter=lambda t: float(t) / 1000,
        has_header=True,
    )
    s, comments = reader(data_folder / "timestamped_float_comments.csv")
    _test_timestamped_float_comments_content(s, comments)


def test__csv_reader__timestamped_float_float(data_folder: pl.Path) -> None:
    reader = CsvReader(
        t_column="t_ms",
        y_column="p",
        delimiter=";",
        t_converter="float",
        has_header=True,
    )
    s, _ = reader(data_folder / "timestamped_float_comments.csv")
    assert s.t.tolist() == [1.5, 3]


def test__csv_reader__timestamped_time(data_folder: pl.Path) -> None:
    reader = CsvReader(
        t_column=0,
        y_column=3,
        delimiter=",",
        t_converter="%H:%M:%S.%f",
        has_header=False,
    )
    s, comments = reader(data_folder / "timestamped_dt_noheader.csv")
    _test_timestamped_dt_noheader_content(s, comments)


def test__csv_reader__non_timestamped(data_folder: pl.Path) -> None:
    reader = CsvReader(
        t_column=None,
        fs=1000,
        y_column=0,
    )
    s, comments = reader(data_folder / "physionet_aac_0027_10000_y.csv")
    assert s.fs == 1000
    assert len(s.y) == 10_000
    assert not comments


def test__csv_reader__invalid() -> None:
    with pytest.raises(ValueError):
        # No fs
        CsvReader(
            t_column=None,
            y_column=1,
        )
    with pytest.raises(ValueError):
        # has_header not set
        CsvReader(
            t_column="t",
            y_column="p",
        )
    with pytest.warns():
        # Ignored fs
        CsvReader(
            t_column=0,
            y_column=1,
            fs=100,
        )


def _test_timestamped_float_comments_content(s: Signal, comments: list[str]) -> None:
    assert s.t.tolist() == [1.5 / 1000, 3 / 1000]
    assert s.y.tolist() == [100.5, 150]
    assert comments == ["sensor: Test", "patient: 1"]


def _test_timestamped_dt_noheader_content(s: Signal, comments: list[str]) -> None:
    assert s.t.tolist() == [50969.457, 50969.458, 50969.459, 50969.46]
    assert s.y.tolist() == [619, 618, 617, 617]
    assert not comments


def test__to_csv(simple_signal: Signal, out_folder: pl.Path) -> None:
    paths = to_csv(out_folder / "to_csv.csv", simple_signal, delimiter=",")
    expected_paths = {
        out_folder / "to_csv.csv",
        out_folder / "to_csv.chpoints.csv",
        out_folder / "to_csv.marks.a.csv",
        out_folder / "to_csv.slices.a.csv",
        out_folder / "to_csv.meta.csv",
    }
    assert paths == expected_paths
    for p in expected_paths:
        assert p.exists()
