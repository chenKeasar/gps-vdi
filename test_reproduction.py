"""Regression test: gpsvdi reproduces the features stored by the production pipeline.

Run with:  pytest test_reproduction.py -q

Reads the deposited data folder, so point GPSVDI_DATA_ROOT at whatever fetch_data.py
populated. The test is skipped, not failed, when the data is not there.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gpsvdi import extract_trap_features, FEATURE_SET

DATA = Path(os.environ.get("GPSVDI_DATA_ROOT", "data"))

# Segmentation must be bit-identical. The other tolerances are set by the reference, not by
# the computation: the deposited table stores some rows at full float precision and others at
# about nine significant digits, which bounds any comparison against it at roughly 1e-8 for
# the colour and size features. The SVD ratios additionally differ in their last digits
# because LAPACK implementations do.
TOL = {"log_insect_size": 1e-7, "rb_ratio": 1e-7, "rg_ratio": 1e-7,
       "svd_ratio01": 1e-4, "svd_ratio12": 1e-4}

N_BOXES = 30


@pytest.fixture(scope="module")
def annotated():
    insects = DATA / "insects.csv"
    reference = DATA / "field_traps.csv"
    if not (insects.exists() and reference.exists()):
        pytest.skip(f"deposit tables not found under {DATA}; run fetch_data.py first")
    df = (pd.read_csv(insects)
            .drop_duplicates("box_id")
            .merge(pd.read_csv(reference, low_memory=False)
                     .dropna(subset=["box_id"]).drop_duplicates("box_id"),
                   on="box_id", suffixes=("", "_stored")))
    df["trap_file"] = df["parent_image_file_name"]
    return df


def test_features_match_the_stored_values(annotated):
    traps = {f.name: f for f in (DATA / "40traps").glob("*.jpg")}
    if not traps:
        pytest.skip(f"no trap scans under {DATA / '40traps'}; run fetch_data.py first")

    trap_file = next(t for t in annotated["trap_file"].value_counts().index if t in traps)
    sub = annotated[annotated["trap_file"] == trap_file].head(N_BOXES)
    boxes = [(r.box_id, r.x, r.y, r.w, r.h) for r in sub.itertuples()]
    got = extract_trap_features(traps[trap_file], boxes, sub["pixels_per_inch"].iloc[0])
    joined = sub.set_index("box_id").join(got, rsuffix="_new")

    # Segmentation must be bit-identical, not merely close.
    assert (joined["insect_size[pixels]"] == joined["insect_size[pixels]_new"]).all()

    for feature in FEATURE_SET:
        old = joined[feature].astype(float)
        new = joined[feature + "_new"].astype(float)
        keep = old.notna() & new.notna()
        assert keep.sum() > 0, f"nothing to compare for {feature}"
        assert np.abs(old[keep] - new[keep]).max() < TOL[feature], feature
