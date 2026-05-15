"""Scoring primitives — splicing, oncosplice, epistasis, fingerprints."""
from .splicing import (
    extract_site_table,
    classify_missplicing,
    DEFAULT_DELTA_THRESHOLD,
)
from .epistasis import (
    compute_site_residuals,
    compute_site_residuals_multi,
    classify_pair,
    DEFAULT_RESIDUAL_THRESHOLD,
)
from .oncosplice import oncosplice_score, OncospliceScore
from .fingerprint import (
    splicing_outcome_fingerprint,
    splicing_outcome_hash,
)

__all__ = [
    "extract_site_table",
    "classify_missplicing",
    "compute_site_residuals",
    "compute_site_residuals_multi",
    "classify_pair",
    "oncosplice_score",
    "OncospliceScore",
    "splicing_outcome_fingerprint",
    "splicing_outcome_hash",
    "DEFAULT_DELTA_THRESHOLD",
    "DEFAULT_RESIDUAL_THRESHOLD",
]
