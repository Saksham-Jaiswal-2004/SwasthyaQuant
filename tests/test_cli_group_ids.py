import numpy as np
import pandas as pd

from qheart.cli import _feature_group_ids


def test_feature_group_ids_identify_identical_rows():
    df = pd.DataFrame(
        {
            "age": [10000, 10000, 12000],
            "height": [170, 170, 180],
            "weight": [70, 70, 80],
            "ap_hi": [120, 120, 130],
            "ap_lo": [80, 80, 85],
            "smoke": [0, 0, 1],
            "alco": [0, 0, 0],
            "active": [1, 1, 1],
            "gender": [1, 1, 2],
            "cholesterol": [1, 1, 2],
            "gluc": [1, 1, 1],
            "cardio": [0, 1, 1],
        }
    )

    groups = _feature_group_ids(df)

    assert groups.shape == (3,)
    assert groups[0] == groups[1]
    assert groups[0] != groups[2]


def test_feature_group_ids_ignore_target():
    df = pd.DataFrame(
        {
            "age": [10000, 10000],
            "height": [170, 170],
            "weight": [70, 70],
            "ap_hi": [120, 120],
            "ap_lo": [80, 80],
            "smoke": [0, 0],
            "alco": [0, 0],
            "active": [1, 1],
            "gender": [1, 1],
            "cholesterol": [1, 1],
            "gluc": [1, 1],
            "cardio": [0, 1],
        }
    )

    groups = _feature_group_ids(df)

    # Target disagreement must not create separate groups.
    assert np.array_equal(groups, [groups[0], groups[0]])