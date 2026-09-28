"""Shared fixtures. Also auto-skips tests whose optional deps are absent, so
`pytest` is green on a fresh clone with only requirements.txt installed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tests.synthetic import make_raw, write_raw

RAW_CSV = Path("data/raw/cardio_train.csv")


def _have(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def pytest_collection_modifyitems(config, items):
    skip_q = pytest.mark.skip(reason="quantum deps not installed (make setup-quantum)")
    skip_d = pytest.mark.skip(reason=f"{RAW_CSV} not present (make data)")
    skip_s = pytest.mark.skip(reason="scikit-learn not installed (make setup)")
    quantum_ok = _have("pennylane") and _have("qiskit")
    data_ok = RAW_CSV.exists()
    sklearn_ok = _have("sklearn")
    for item in items:
        if "quantum" in item.keywords and not quantum_ok:
            item.add_marker(skip_q)
        if "needs_data" in item.keywords and not data_ok:
            item.add_marker(skip_d)
        if "needs_sklearn" in item.keywords and not sklearn_ok:
            item.add_marker(skip_s)


@pytest.fixture
def raw_df():
    return make_raw()


@pytest.fixture
def raw_csv(tmp_path):
    p = tmp_path / "cardio_train.csv"
    write_raw(p)
    return p


@pytest.fixture
def clean_df(raw_csv):
    from qheart.data.loaders import load
    return load("cardio_70000", raw_csv)
