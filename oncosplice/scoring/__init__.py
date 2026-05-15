"""Scoring primitives — splicing, oncosplice, epistasis, fingerprints."""
from .epistasis import (
    DEFAULT_RESIDUAL_THRESHOLD,
    classify_pair,
    compute_site_residuals,
    compute_site_residuals_multi,
)
from .fingerprint import (
    splicing_outcome_fingerprint,
    splicing_outcome_hash,
)
from .oncosplice import OncospliceScore, oncosplice_score
from .splicing import (
    DEFAULT_DELTA_THRESHOLD,
    classify_missplicing,
    extract_site_table,
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
