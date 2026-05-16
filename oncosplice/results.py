"""
Result containers — single-variant, double-variant, N-variant, protein library.

These are pure dataclasses; the engine builds and returns them. They expose
``.to_dict()`` for JSON serialization, ``.summary()`` for one-line stats, and
plotting hooks (``.plot_*``) that delegate to :mod:`oncosplice.viz`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# ----------------------------------------------------------------------------
# Common pieces
# ----------------------------------------------------------------------------

@dataclass
class MissplicingProfile:
    """Per-position changes in splicing probability for one variant context."""
    missed_donors:        Dict[int, dict]
    missed_acceptors:     Dict[int, dict]
    discovered_donors:    Dict[int, dict]
    discovered_acceptors: Dict[int, dict]

    @property
    def n_events(self) -> int:
        return (len(self.missed_donors) + len(self.missed_acceptors)
                + len(self.discovered_donors) + len(self.discovered_acceptors))

    @property
    def max_abs_delta(self) -> float:
        deltas = []
        for bucket in (self.missed_donors, self.missed_acceptors,
                       self.discovered_donors, self.discovered_acceptors):
            deltas.extend(abs(v["delta"]) for v in bucket.values())
        return max(deltas) if deltas else 0.0

    def to_dict(self) -> dict:
        return {
            "missed_donors":        self.missed_donors,
            "missed_acceptors":     self.missed_acceptors,
            "discovered_donors":    self.discovered_donors,
            "discovered_acceptors": self.discovered_acceptors,
        }

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for bucket_name, bucket in [
            ("missed_donor", self.missed_donors),
            ("missed_acceptor", self.missed_acceptors),
            ("discovered_donor", self.discovered_donors),
            ("discovered_acceptor", self.discovered_acceptors),
        ]:
            for pos, vals in bucket.items():
                rows.append({"position": pos, "event_type": bucket_name, **vals})
        if not rows:
            return pd.DataFrame(columns=["position", "event_type", "ref_prob", "var_prob", "delta"])
        return pd.DataFrame(rows).sort_values("position").reset_index(drop=True)


@dataclass
class SiteEpistasis:
    """One row of the per-site residual table for a double variant."""
    position: int
    site_type: str
    annotated: bool
    ref: float
    mut1: float
    mut2: float
    event: float
    expected: float
    residual: float
    classification: str  # one of CATEGORIES in scoring.epistasis

    def to_dict(self) -> dict:
        return {
            "position": int(self.position),
            "site_type": self.site_type,
            "annotated": bool(self.annotated),
            "ref": float(self.ref),
            "mut1": float(self.mut1),
            "mut2": float(self.mut2),
            "event": float(self.event),
            "expected": float(self.expected),
            "residual": float(self.residual),
            "classification": self.classification,
        }


# ----------------------------------------------------------------------------
# Single-variant result
# ----------------------------------------------------------------------------

@dataclass
class SingleVariantResult:
    """Everything we computed for one mutation."""

    mut_id: str
    gene: str
    transcript_id: str
    splicing_engine: str
    central_position: int

    # Splicing-level outputs
    missplicing: MissplicingProfile
    max_splicing_delta: float

    # Per-isoform reconstruction (Oncosplice paper's transcriptome step)
    isoforms: pd.DataFrame
    """Columns: isoform_id, prevalence, splicing_changes, oncosplice_score, percentile, ..."""

    # Aggregated functional-divergence score
    oncosplice_score: float
    percentile: float

    # Underlying splicing landscape (long format) for plotting / reuse
    site_table: pd.DataFrame

    # Reference protein/mRNA the analysis was done against
    reference_protein: str
    reference_mrna: str

    # ----- methods -----

    def summary(self) -> dict:
        return {
            "mut_id":             self.mut_id,
            "gene":               self.gene,
            "splicing_engine":    self.splicing_engine,
            "n_isoforms":         len(self.isoforms),
            "max_splicing_delta": float(self.max_splicing_delta),
            "n_missplicing":      self.missplicing.n_events,
            "oncosplice_score":   float(self.oncosplice_score),
            "percentile":         float(self.percentile),
        }

    def to_dict(self) -> dict:
        return {
            **self.summary(),
            "missplicing": self.missplicing.to_dict(),
            "isoforms": self.isoforms.to_dict(orient="records"),
        }

    def plot_missplicing(self, **kwargs):
        from .viz import plot_single_missplicing
        return plot_single_missplicing(self, **kwargs)


# ----------------------------------------------------------------------------
# Double-variant result
# ----------------------------------------------------------------------------

@dataclass
class DoubleVariantResult:
    """Everything we computed for a pair of mutations."""

    mut1_id: str
    mut2_id: str
    epistasis_id: str
    gene: str
    transcript_id: str
    splicing_engine: str
    distance: int
    central_position: int

    # Pair-level call (see scoring.epistasis.CATEGORIES)
    pair_classification: str
    score_residual: float     # event_score - (mut1_score + mut2_score - ref_score) on Oncosplice scores

    # Per-context Oncosplice scores
    oncosplice_scores: Dict[str, float]  # {ref, mut1, mut2, event}
    oncosplice_percentiles: Dict[str, float]

    # Per-site epistasis residuals
    site_residuals: pd.DataFrame
    """Columns: position, site_type, annotated, ref, mut1, mut2, event,
    expected, residual, classification."""

    # Pair-level numerical summary
    epistasis_summary: dict

    # Underlying splicing landscape (long format), four contexts
    site_table: pd.DataFrame

    # Per-isoform breakdown of the joint event
    isoforms_event: pd.DataFrame

    # Per-isoform breakdowns for the constituent variants (for completeness)
    isoforms_mut1: Optional[pd.DataFrame] = None
    isoforms_mut2: Optional[pd.DataFrame] = None

    reference_protein: str = ""

    # Convenience attributes set at construction time
    @property
    def is_epistatic(self) -> bool:
        return self.pair_classification != "non-epistatic"

    @property
    def max_abs_residual(self) -> float:
        return float(self.epistasis_summary.get("max_abs_residual", 0.0))

    def summary(self) -> dict:
        return {
            "epistasis_id":         self.epistasis_id,
            "gene":                 self.gene,
            "distance":             int(self.distance),
            "splicing_engine":      self.splicing_engine,
            "pair_classification":  self.pair_classification,
            "max_abs_residual":     self.max_abs_residual,
            "n_del_syn_sites":      self.epistasis_summary.get("n_del_syn", 0),
            "n_cryp_syn_sites":     self.epistasis_summary.get("n_cryp_syn", 0),
            "n_rescue_sites":       self.epistasis_summary.get("n_rescue", 0),
            "n_cryp_rescue_sites":  self.epistasis_summary.get("n_cryp_rescue", 0),
            "max_rescue_residual":  self.epistasis_summary.get("max_rescue_residual", 0.0),
            "max_synergy_residual": self.epistasis_summary.get("max_synergy_residual", 0.0),
            "score_ref":            self.oncosplice_scores.get("ref", 0.0),
            "score_mut1":           self.oncosplice_scores.get("mut1", 0.0),
            "score_mut2":           self.oncosplice_scores.get("mut2", 0.0),
            "score_event":          self.oncosplice_scores.get("event", 0.0),
            "score_residual":       float(self.score_residual),
        }

    def epistatic_sites(self) -> pd.DataFrame:
        """Return only the splice sites flagged as syn/ant."""
        if self.site_residuals.empty:
            return self.site_residuals
        return self.site_residuals[
            self.site_residuals.classification != "non-epistatic"
        ].reset_index(drop=True)

    def to_dict(self) -> dict:
        return {
            **self.summary(),
            "site_residuals":   self.site_residuals.to_dict(orient="records"),
            "epistasis_summary": self.epistasis_summary,
        }

    def plot_residuals(self, **kwargs):
        from .viz import plot_pair_residuals
        return plot_pair_residuals(self, **kwargs)

    def plot_landscape(self, **kwargs):
        from .viz import plot_pair_landscape
        return plot_pair_landscape(self, **kwargs)

    def plot_summary(self, **kwargs):
        """Combined view: top = per-position landscape across contexts;
        middle = expected vs observed at each splice site; bottom = residual bars."""
        from .viz import plot_pair_summary
        return plot_pair_summary(self, **kwargs)

    def plot_case_study(self, **kwargs):
        """One-page mechanistic case-study figure for this pair.

        Three stacked panels: (1) gene-context strip with variants + annotated
        splice sites, (2) horizontal probability bars for each affected splice
        site across ref/mut1/mut2/event with classification tinting,
        (3) summary banner with classification badge + auto-generated
        mechanistic interpretation.

        Returns ``(fig, (ax_top, ax_main, ax_bottom))``.
        """
        from .viz import plot_pair_case_study
        return plot_pair_case_study(self, **kwargs)

    def protein_library(self) -> "ProteinLibrary":
        """Wrap the event-context isoforms as a typed :class:`ProteinLibrary`.

        Use ``DoubleVariantResult.isoforms_event`` directly for the raw
        DataFrame; this wrapper adds ``.to_fasta()`` and ``.filter()``.
        """
        return ProteinLibrary(
            reference_protein=self.reference_protein,
            isoforms=self.isoforms_event if self.isoforms_event is not None else pd.DataFrame(),
            context=f"event ({self.epistasis_id})",
        )


# ----------------------------------------------------------------------------
# N-variant result (3 or more mutations on the same gene)
# ----------------------------------------------------------------------------

@dataclass
class MultiVariantResult:
    """Splicing-residual breakdown for an N-variant construct (N ≥ 2)."""

    construct_id: str           # '|'-joined canonical mutation IDs
    mut_ids: List[str]
    gene: str
    transcript_id: str
    splicing_engine: str
    n_variants: int
    central_position: int

    pair_classification: str    # see scoring.epistasis.CATEGORIES

    site_residuals: pd.DataFrame
    """Long-format: per (position, site_type) row with ref, mut1..mutN, event,
    expected, residual, classification."""

    epistasis_summary: dict
    site_table: pd.DataFrame

    # Optional protein-level data (skipped by scan paths; populated by
    # :meth:`OncospliceEngine.analyze_multi` when ``protein=True``).
    isoforms_event: Optional[pd.DataFrame] = None
    oncosplice_score_event: Optional[float] = None
    reference_protein: str = ""

    @property
    def is_epistatic(self) -> bool:
        return self.pair_classification != "non-epistatic"

    @property
    def max_abs_residual(self) -> float:
        return float(self.epistasis_summary.get("max_abs_residual", 0.0))

    @property
    def epistasis_id(self) -> str:
        """Alias for ``construct_id`` — matches DoubleVariantResult naming."""
        return self.construct_id

    def summary(self) -> dict:
        return {
            "construct_id":         self.construct_id,
            "gene":                 self.gene,
            "n_variants":           int(self.n_variants),
            "splicing_engine":      self.splicing_engine,
            "pair_classification":  self.pair_classification,
            "max_abs_residual":     self.max_abs_residual,
            "n_del_syn_sites":      self.epistasis_summary.get("n_del_syn", 0),
            "n_cryp_syn_sites":     self.epistasis_summary.get("n_cryp_syn", 0),
            "n_rescue_sites":       self.epistasis_summary.get("n_rescue", 0),
            "n_cryp_rescue_sites":  self.epistasis_summary.get("n_cryp_rescue", 0),
            "oncosplice_score_event": self.oncosplice_score_event,
        }

    def epistatic_sites(self) -> pd.DataFrame:
        if self.site_residuals.empty:
            return self.site_residuals
        return self.site_residuals[
            self.site_residuals.classification != "non-epistatic"
        ].reset_index(drop=True)

    def to_dict(self) -> dict:
        return {
            **self.summary(),
            "mut_ids":           list(self.mut_ids),
            "site_residuals":    self.site_residuals.to_dict(orient="records"),
            "epistasis_summary": self.epistasis_summary,
        }

    def protein_library(self) -> "ProteinLibrary":
        return ProteinLibrary(
            reference_protein=self.reference_protein,
            isoforms=self.isoforms_event if self.isoforms_event is not None else pd.DataFrame(),
            context=f"event ({self.construct_id})",
        )


# ----------------------------------------------------------------------------
# Protein isoform library
# ----------------------------------------------------------------------------

@dataclass
class ProteinLibrary:
    """Collection of predicted alternative-isoform proteins for a variant context.

    The ``isoforms`` DataFrame mirrors the columns produced by the engine:
    ``isoform_id``, ``prevalence``, ``splicing_changes``, ``oncosplice_score``,
    ``percentile``, ``n_deletions``, ``n_insertions``, ``variant_protein`` plus
    SpliceSimulator metadata (``es`` exon-skipping, ``ir`` intron-retention,
    ``pes`` partial-exon-skip, ``pir`` partial-intron-retention, ``ne``
    new-exon).
    """
    reference_protein: str
    isoforms: pd.DataFrame
    context: str = "event"

    @property
    def n_isoforms(self) -> int:
        return int(len(self.isoforms))

    @property
    def total_prevalence(self) -> float:
        if self.isoforms.empty or "prevalence" not in self.isoforms.columns:
            return 0.0
        return float(self.isoforms.prevalence.fillna(0).sum())

    def filter(self, *, min_prevalence: float = 0.0,
               min_score: float = 0.0,
               distinct_only: bool = False) -> "ProteinLibrary":
        """Return a new ProteinLibrary keeping only isoforms that pass filters."""
        df = self.isoforms
        if df.empty:
            return ProteinLibrary(self.reference_protein, df, self.context)
        mask = pd.Series(True, index=df.index)
        if "prevalence" in df.columns:
            mask &= df.prevalence.fillna(0) >= min_prevalence
        if "oncosplice_score" in df.columns:
            mask &= df.oncosplice_score.fillna(0) >= min_score
        if distinct_only and "variant_protein" in df.columns:
            mask &= df.variant_protein != self.reference_protein
        return ProteinLibrary(self.reference_protein, df[mask].reset_index(drop=True), self.context)

    def to_fasta(self, path: str | Path, include_reference: bool = True) -> Path:
        """Write the library to a FASTA file. Returns the path written."""
        path = Path(path)
        lines: List[str] = []
        if include_reference and self.reference_protein:
            lines.append(f">REF | context={self.context}")
            lines.extend(_wrap_fasta(self.reference_protein))
        if not self.isoforms.empty and "variant_protein" in self.isoforms.columns:
            for _, row in self.isoforms.iterrows():
                iso_id = row.get("isoform_id", "")
                prev   = row.get("prevalence", "")
                score  = row.get("oncosplice_score", "")
                lines.append(
                    f">{iso_id} | context={self.context} | prevalence={prev} | "
                    f"oncosplice_score={score}"
                )
                lines.extend(_wrap_fasta(str(row["variant_protein"] or "")))
        path.write_text("\n".join(lines) + "\n")
        return path

    def summary(self) -> dict:
        return {
            "context":           self.context,
            "n_isoforms":        self.n_isoforms,
            "total_prevalence":  self.total_prevalence,
            "reference_length":  len(self.reference_protein),
        }


def _wrap_fasta(seq: str, width: int = 60) -> List[str]:
    return [seq[i:i + width] for i in range(0, len(seq), width)]
