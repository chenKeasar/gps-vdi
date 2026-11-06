"""gpsvdi - Grid Pseudo-Species Visual Diversity Index.

Feature extraction that runs from trap scans alone, without STARdbi.
"""

from .extract import FEATURE_SET, extract_trap_features

__all__ = ["extract_trap_features", "FEATURE_SET"]
