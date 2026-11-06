"""Compute the five GPS-VDI visual features directly from a trap scan.

This is the step that removes the dependency on precalculated features: given a
trap image and the bounding boxes of its detected arthropods, it segments each
arthropod from the trap background, measures its projected size, and derives the
colour and shape descriptors of manuscript section 2.3.2.

Validated against STARdbi's stored values: colour and SVD features agree to
floating-point noise, and segmentation pixel counts agree exactly.
"""

import json

import cv2
import numpy as np
import pandas as pd

from .features import compute_box_biodiversity_features
from .segmentation import InsectCoverage

FEATURE_SET = ["log_insect_size", "rb_ratio", "rg_ratio", "svd_ratio12", "svd_ratio01"]


def _clip_box(x, y, w, h, img_w, img_h):
    """Clip a box to the image, mirroring InsectCoverage.make_masks."""
    if x + w > img_w:
        w = img_w - x
    if y + h > img_h:
        h = img_h - y
    return max(0, x), max(0, y), w, h


def extract_trap_features(trap_path, boxes, ppi):
    """Features for every box on one trap.

    Parameters
    ----------
    trap_path : str or Path   path to the trap scan (RGB jpg).
    boxes     : sequence of (box_id, x, y, w, h) in pixel coordinates.
    ppi       : scan resolution, as stored by STARdbi (e.g. "1200, 1200" or 1200).

    Returns
    -------
    pandas.DataFrame indexed by box_id with the seven per-box features; the five
    used by GPS-VDI are listed in ``FEATURE_SET``.
    """
    boxes = list(boxes)
    coords = [(int(x), int(y), int(w), int(h)) for _, x, y, w, h in boxes]
    coverage = InsectCoverage(str(trap_path), coords)
    coverage.make_masks()
    img_h, img_w = coverage.height, coverage.width

    records = []
    for (box_id, x, y, w, h), mask in zip(boxes, coverage.masks):
        x, y, w, h = _clip_box(int(x), int(y), int(w), int(h), img_w, img_h)
        crop_bgr = coverage.image[y:y + h, x:x + w]
        insect_size = int(np.sum(mask))

        grey = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(float)
        svd_grayscale = json.dumps(np.linalg.svd(grey, compute_uv=False).tolist())
        svd_mask = json.dumps(
            np.linalg.svd(mask.astype(float), compute_uv=False).tolist())

        feats = compute_box_biodiversity_features(
            crop_bgr=crop_bgr, insect_size=insect_size, ppi_raw=ppi,
            svd_mask_json=svd_mask, svd_grayscale_json=svd_grayscale)
        feats["box_id"] = box_id
        feats["insect_size[pixels]"] = insect_size
        records.append(feats)

    return pd.DataFrame.from_records(records).set_index("box_id")
