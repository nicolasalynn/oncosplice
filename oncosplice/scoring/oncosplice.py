"""
Functional-divergence scoring (the *Oncosplice score*).

Implements the published algorithm:
- pairwise-align reference and variant proteins with mismatches re-cast as
  deletions
- transform a Rate4Site conservation vector with a parabolic window of length
  ``W`` and exponential scaling (so highly-conserved residues weigh more)
- penalize each indel by ``max(1, indel_length / W) * smoothed_conservation``
  over the modified positions
- final score = max of the resulting vector; percentile rank of that max
  within the smoothed conservation distribution.

This is a thin reorganization of ``geney.oncosplice.Oncosplice`` so that the
oncosplice package is self-contained and readable.  The math is identical.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .._geney_compat import Oncosplice as _GeneyOncosplice


@dataclass(frozen=True)
class OncospliceScore:
    """Output of :func:`oncosplice_score`."""
    score: float
    percentile: float
    n_deletions: int
    n_insertions: int
    n_modified_positions: int
    aligned_reference: str
    aligned_variant: str

    def as_dict(self) -> dict:
        return {
            "score": float(self.score),
            "percentile": float(self.percentile),
            "n_deletions": int(self.n_deletions),
            "n_insertions": int(self.n_insertions),
            "n_modified_positions": int(self.n_modified_positions),
        }


def oncosplice_score(
    reference_protein: str,
    variant_protein: str,
    conservation_vector: Optional[np.ndarray] = None,
    *,
    window_length: int = 13,
) -> OncospliceScore:
    """Compute the Oncosplice functional-divergence score for one variant protein.

    Parameters
    ----------
    reference_protein, variant_protein
        Amino-acid strings (no asterisks). Empty ``variant_protein`` is allowed
        and triggers the published fallback to a single-residue alignment.
    conservation_vector
        Per-residue Rate4Site conservation scores (length matches
        ``reference_protein``). If ``None``, an all-ones vector is used,
        which collapses the score to a length-weighted indel penalty.
    window_length
        Smoothing window for the parabolic transform (paper default 13).
    """
    ref = reference_protein or ""
    var = variant_protein or ""
    if conservation_vector is None:
        conservation_vector = np.ones(len(ref) or 1, dtype=float)
    else:
        conservation_vector = np.asarray(conservation_vector, dtype=float)

    onco = _GeneyOncosplice(
        reference_protein=ref,
        variant_protein=var,
        conservation_vector=conservation_vector,
        window_length=window_length,
    )
    return OncospliceScore(
        score=float(onco.score),
        percentile=float(onco.percentile),
        n_deletions=len(onco.deletions),
        n_insertions=len(onco.insertions),
        n_modified_positions=int(onco.modified_positions.sum()),
        aligned_reference=str(onco.alignment.seqA),
        aligned_variant=str(onco.alignment.seqB),
    )


def aggregate_isoform_scores(scores_with_prevalence: list[Tuple[float, float]]) -> float:
    """Weakest-link aggregate across isoforms of one transcript.

    The published rule is *weakest-link*: take the prevalence-weighted mean of
    the per-isoform scores within a transcript.  Across multiple transcripts
    of a gene, take the maximum.  This function performs the within-transcript
    aggregation; cross-transcript aggregation is a simple ``max`` at the
    caller.

    ``scores_with_prevalence`` is a list of ``(score, prevalence)`` tuples
    where prevalences should sum to 1 (we renormalize defensively).
    """
    if not scores_with_prevalence:
        return 0.0
    arr = np.array(scores_with_prevalence, dtype=float)
    scores, weights = arr[:, 0], arr[:, 1]
    if weights.sum() <= 0:
        return float(scores.mean())
    weights = weights / weights.sum()
    return float((scores * weights).sum())
