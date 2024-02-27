import pathlib as pl
from typing import Any

import numpy as np
import pytest

from bpwave import ChPoints, CpIndices, Signal


@pytest.fixture(scope="session")
def data_folder() -> pl.Path:
    return pl.Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def out_folder() -> pl.Path:
    out = pl.Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    return out


@pytest.fixture(scope="session")
def bp_signal_data(data_folder: pl.Path) -> dict[str, Any]:
    test_data: dict[str, Any] = {"fs": 1000}
    for p in data_folder.glob("physionet_aac_0027_10000_*.csv"):
        test_data[p.stem.replace("physionet_aac_0027_10000_", "")] = np.loadtxt(
            p
        ).astype(float if p.stem.endswith("_y") else int)
    return test_data


@pytest.fixture(scope="function")
def bp_signal(bp_signal_data: dict[str, Any]) -> Signal:
    return Signal(
        y=bp_signal_data["y"],
        fs=bp_signal_data["fs"],
        chpoints=ChPoints(
            indices=[
                CpIndices(onset=o, sys_peak=s, refl_peak=r, dicr_notch=dn, dicr_peak=dp)
                for (o, s, r, dn, dp) in zip(
                    bp_signal_data["onset"],
                    bp_signal_data["sys_peak"],
                    bp_signal_data["refl_peak"],
                    bp_signal_data["dicr_notch"],
                    bp_signal_data["dicr_peak"],
                )
            ],
            alg="manual",
            version="0.1.0",
            params={},
        ),
        label="AAC27",
        meta={"source": "PhysioNet"},
    )
