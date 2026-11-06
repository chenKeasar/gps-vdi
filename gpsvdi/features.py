"""
Per-box biodiversity feature extraction.

Vendored from STARdbi (helpers/biodiversity_features.py); only the config import
was rewritten. Formulas unchanged.

numpy + cv2 only — no pandas, no ``BiodiversityUtils``, no colab. Every feature
is computed defensively: a feature whose own computation fails (``ValueError``,
divide-by-zero, NaN/Inf, missing/short data) comes back as ``None`` so the caller
can store ``NULL`` for that one cell only and never lose the whole row (spec D2).

Math mirrors the customer's offline pipeline exactly:

  rb/rg/bg_ratio        BiodiversityUtils/get_datframe_from_insect_images._get_image_data
  log_insect_size       BiodiversityUtils/Engineered_features_utils/get_log_insect_size
  svd_normalized_entropy BiodiversityUtils/Engineered_features_utils/get_svd_normalized_entropy
  svd_ratio01 / 12      BiodiversityUtils/Engineered_features_utils/get_svd_ratio  (double log, D1)

All tunables come from ``helpers.biodiversity_config``.
"""

import ast
import json
import math

import cv2
import numpy as np

from . import _config as cfg


# ───────────────────────────────── small helpers ──────────────────────────────────

def _clean(value):
    """Return a finite float, or ``None`` for None/NaN/Inf/non-numeric (MySQL DOUBLE
    cannot hold NaN/Inf — store NULL instead)."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _to_float_list(value):
    """Parse a stored SVD value into a list of floats. Accepts the binary column
    (packed little-endian ``float32`` bytes) and, for transition/legacy callers, a
    JSON string or python list. Returns ``[]`` on anything unparseable."""
    if value is None:
        return []
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            return np.frombuffer(value, dtype='<f4').tolist()
        except (ValueError, TypeError):
            return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return []
    try:
        return [float(v) for v in value]
    except (TypeError, ValueError):
        return []


def parse_ppi(x):
    """pixels-per-inch from ``TrapImage.pixels_per_inch`` (raw, e.g. ``"(1200, 1200)"``,
    ``"1200"``, an int, or a tuple). Mirrors ``BiodiversityUtils.parse_ppi`` but returns
    ``None`` instead of raising. The first axis value is used (square pixels assumed)."""
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return None if math.isnan(x) else int(x)
    if isinstance(x, (tuple, list)):
        return int(x[0]) if x else None
    s = str(x).strip()
    if not s:
        return None
    words = s.split(',')
    try:
        if len(words) >= 2:
            return int(words[0].split('(')[-1].strip())
        return int(s.strip('() '))
    except ValueError:
        return None


# ─────────────────────────────────── features ─────────────────────────────────────

def color_ratios(crop_bgr):
    """(rb, rg, bg) channel-sum ratios on the box crop.

    The customer's crops were encoded BGR (cv2) then read back as PIL RGB, so we
    convert ``COLOR_BGR2RGB`` first, then reproduce ``_get_image_data``:
        rb = (ΣR + 1) / (ΣB + 1),  rg = (ΣR + 1) / (ΣG + 1),  bg = (ΣB + 1) / (ΣG + 1).
    """
    if crop_bgr is None or getattr(crop_bgr, 'size', 0) == 0:
        return None, None, None
    try:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        r = float(np.sum(rgb[:, :, 0]))
        g = float(np.sum(rgb[:, :, 1]))
        b = float(np.sum(rgb[:, :, 2]))
    except (cv2.error, IndexError, ValueError):
        return None, None, None
    off = cfg.COLOR_RATIO_OFFSET
    rb = _clean((r + off) / (b + off)) if (b + off) != 0 else None
    rg = _clean((r + off) / (g + off)) if (g + off) != 0 else None
    bg = _clean((b + off) / (g + off)) if (g + off) != 0 else None
    return rb, rg, bg


def log_insect_size(insect_size_px, ppi_raw):
    """ln( (insect_size + 1) / (ppi / 1200)^2 ) — ``get_log_insect_size``."""
    ppi = parse_ppi(ppi_raw)
    if ppi is None or insect_size_px is None:
        return None
    resolution = ppi / cfg.PPI_REFERENCE
    if resolution == 0:
        return None
    size = (insect_size_px + cfg.SIZE_PIXEL_OFFSET) / (resolution ** 2)
    if size <= 0:
        return None
    return _clean(math.log(size))


def svd_normalized_entropy(svd_mask_json):
    """Shannon entropy of the (filtered, sum-normalised) ``svd_mask`` singular values.

    Reproduces the *active* ``get_svd_normalized_entropy``: keep singular values
    ``>= SVD_MIN_SINGULAR_VALUE``, normalise to a probability vector (÷ sum), then
    ``-Σ p·ln p`` over ``p > 0``. (It is *not* divided by ln(n) — that variant is
    commented out in the notebook.)
    """
    vals = [v for v in _to_float_list(svd_mask_json) if v >= cfg.SVD_MIN_SINGULAR_VALUE]
    if not vals:
        return None
    arr = np.asarray(vals, dtype=float)
    total = arr.sum()
    if total <= 0:
        return None
    p = arr / total
    p = p[p > 0]
    if p.size == 0:
        return None
    return _clean(float(-(p * np.log(p)).sum()))


def svd_ratio_log(svd_grayscale_json, i, j):
    """Singular-value ratio feature on the filtered ``svd_grayscale`` list.

    ``get_svd_ratio`` returns ``log10(s_i / s_j)``; ``merge_dataframes`` wraps that in
    another ``np.log`` (the double log, D1). Indices are into the list filtered to
    ``>= SVD_MIN_SINGULAR_VALUE``. Returns ``None`` when an index is out of range, the
    ratio is non-positive, or the (double) log is undefined.
    """
    vals = [v for v in _to_float_list(svd_grayscale_json) if v >= cfg.SVD_MIN_SINGULAR_VALUE]
    if i >= len(vals) or j >= len(vals) or i < 0 or j < 0:
        return None
    numer, denom = vals[i], vals[j]
    if denom == 0:
        return None
    ratio = numer / denom
    if ratio <= 0:
        return None
    log10v = math.log10(ratio)
    if not cfg.SVD_RATIO_DOUBLE_LOG:
        return _clean(log10v)
    if log10v <= 0:                       # ln of a non-positive number is undefined → NULL
        return None
    return _clean(math.log(log10v))


def compute_box_biodiversity_features(crop_bgr, insect_size, ppi_raw,
                                      svd_mask_json, svd_grayscale_json):
    """The 7 per-box features as a dict keyed by ``BIODIVERSITY_FEATURE_COLUMNS``.

    Every feature is isolated: a failure in one yields ``None`` for that key only
    (spec D2). The returned dict always has exactly the 7 keys.
    """
    rb, rg, bg = color_ratios(crop_bgr)
    features = {
        'rb_ratio': rb,
        'rg_ratio': rg,
        'bg_ratio': bg,
        'log_insect_size': log_insect_size(insect_size, ppi_raw),
        'svd_normalized_entropy': svd_normalized_entropy(svd_mask_json),
    }
    for column, (i, j) in cfg.SVD_RATIO_PAIRS.items():
        features[column] = svd_ratio_log(svd_grayscale_json, i, j)
    return features
