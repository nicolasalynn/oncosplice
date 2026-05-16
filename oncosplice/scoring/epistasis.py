"""
Splicing-epistasis classification — 4 crisp mechanism classes (+ non-epistatic).

At each splice site we have four predicted probabilities (all in [0, 1]):
``ref``, ``mut1``, ``mut2``, ``event`` (the joint), and one boolean
``annotated``. We define::

    expected = mut1 + mut2 - ref          # additive null on probability scale
    de       = event - ref                # observed joint shift from WT
    residual = event - expected           # signed excess over additive

Hard prerequisite — WT prediction must match annotation
-------------------------------------------------------

Every classification rule requires the wild-type prediction to agree with the
annotation:

- ``annotated == True``  → must have ``ref ≥ HIGH`` (site is annotated *and*
  the engine predicts it).
- ``annotated == False`` → must have ``ref ≤ LOW``  (site is not annotated
  *and* the engine doesn't predict it).

Sites where the WT-prediction disagrees with the annotation (e.g. annotated
acceptor that the engine misses, or a cryptic site the engine "hallucinates"
in WT) are dropped as non-epistatic without ever consulting the mutations.
This is the engine's own noise filter: we only trust per-pair calls at sites
the engine itself recognises correctly.

Four mechanism classes
----------------------

Given a site that passes the WT-vs-annotation prerequisite, one of four
mutually-exclusive rules can fire:

1. **rescue** — annotated site destroyed by one variant, restored by the joint.

   * ``annotated == True`` and ``ref ≥ HIGH``
   * ``min(mut1, mut2) ≤ ref − HIGH`` (at least one single drops the site by
     ≥ ``HIGH`` — substantial deletion)
   * ``|event − ref| ≤ NEAR_WT`` (joint back near WT)
   * ``rescue_residual = event − min(mut1, mut2) ≥ RES``

2. **cryptic_rescue** — novel site created by one variant, suppressed by the
   joint.

   * ``annotated == False`` and ``ref ≤ LOW``
   * ``max(mut1, mut2) ≥ HIGH`` (at least one single creates a cryptic site)
   * ``event ≤ LOW`` (joint silences it)
   * ``rescue_residual = max(mut1, mut2) − event ≥ RES``

3. **deletion_synergy** — annotated site preserved by each single, destroyed
   only when both co-occur.

   * ``annotated == True`` and ``ref ≥ HIGH``
   * ``min(mut1, mut2) ≥ HIGH`` (each single alone preserves the site)
   * ``ref − event ≥ RES`` (joint drops the site vs WT)
   * ``expected − event ≥ RES`` (joint drops the site vs the additive null —
     this is the *synergy* itself)
   * ``synergy_residual = expected − event``

4. **cryptic_synergy** — novel site requires both variants together.

   * ``annotated == False`` and ``ref ≤ LOW``
   * ``max(mut1, mut2) ≤ LOW`` (no single alone creates anything)
   * ``event ≥ HIGH`` (joint creates a substantial cryptic site)
   * ``event − expected ≥ RES`` (synergy magnitude over additive null)
   * ``synergy_residual = event − expected``

Anything else → **non-epistatic**.

Pair-level aggregation
----------------------

A pair's overall label is the class of the site with the *largest* mechanism
residual (``rescue_residual`` for the rescues, ``synergy_residual`` for the
synergies). Ties are broken by class priority::

    deletion_synergy  >  cryptic_synergy  >  rescue  >  cryptic_rescue
        >  non-epistatic

The full per-site breakdown is always retained in the residuals frame —
``analyze_pair`` returns one row per splice site with the four predictions
and the per-site class.

Thresholds
----------

* ``HIGH_BAND        = 0.50`` — "site present" (includes alt-splice sites).
* ``LOW_BAND         = 0.05`` — "site absent".
* ``RESIDUAL_THRESHOLD = 0.10`` — minimum residual magnitude. Permissive on
  purpose: deletion / cryptic synergies are flagged with residuals as small
  as 0.10. The strict WT-vs-annotation prerequisite is what keeps noise out.
* ``NEAR_WT          = 0.20`` — ``|event − ref|`` tolerance for rescue.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List

import pandas as pd

from .splicing import site_table_wide

# ── canonical thresholds (do not tune lightly — README documents these) ──────
HIGH_BAND               = 0.50   # "site present"  (incl. alt-spliced)
LOW_BAND                = 0.05   # "no site"
RESIDUAL_THRESHOLD      = 0.10   # required residual magnitude (Δ vs expected)
NEAR_WT                 = 0.20   # |event − ref| tolerance for rescue

# legacy threshold names — kept as kwargs aliases for API stability only
DEFAULT_RESIDUAL_THRESHOLD       = RESIDUAL_THRESHOLD
DEFAULT_TOTAL_EFFECT_THRESHOLD   = RESIDUAL_THRESHOLD
DEFAULT_SINGLE_EFFECT_THRESHOLD  = RESIDUAL_THRESHOLD
DEFAULT_RESCUE_PROXIMITY         = NEAR_WT
DEFAULT_ACTIVITY_FLOOR           = 0.0

# Priority for pair-level tie-breaking when residuals match. Listing
# synergies first is biological taste: an emergent joint effect is the more
# striking finding than a rescue when both magnitudes are equal.
CATEGORIES = (
    "deletion_synergy",
    "cryptic_synergy",
    "rescue",
    "cryptic_rescue",
    "non-epistatic",
)
_PRIORITY = {c: i for i, c in enumerate(CATEGORIES)}

# convenience: which sites are "epistatic" (any non-null class)
EPISTATIC = tuple(c for c in CATEGORIES if c != "non-epistatic")


@dataclass(frozen=True)
class SiteResidual:
    """Per-splice-site epistasis breakdown.

    Each row carries four *always-computed* signed residuals, one per class,
    each oriented so that a positive value means "this mechanism is at play":

    * ``rescue_residual           = event − min(mut1, mut2)``
        positive ⇒ joint is above the damaged single (site is restored)
    * ``cryptic_rescue_residual   = max(mut1, mut2) − event``
        positive ⇒ joint is below the cryptic-creating single (cryptic silenced)
    * ``deletion_synergy_residual = expected − event``
        positive ⇒ joint below additive (annotated site destroyed beyond null)
    * ``cryptic_synergy_residual  = event − expected``
        positive ⇒ joint above additive (novel site created beyond null)

    On top of the four class-specific residuals, two signed general residuals
    summarise the per-site epistasis. Both share the convention
    ``+ve ⇒ synergy direction``, ``−ve ⇒ rescue direction``, so they're
    directly comparable:

    * ``residual_vs_expected``
        Classical additive-null residual.
        annotated: ``expected − event`` (positive = below additive).
        cryptic:   ``event − expected`` (positive = above additive).
    * ``residual_vs_individuals``
        Distance past the most-relevant individual single.
        annotated: ``max(min_mut − event, 0) − max(event − max_mut, 0)``
            i.e. positive = past the worst-damaged single, negative = past
            the best-preserved single. Zero when event lies between the two
            singles.
        cryptic:   ``max(event − max_mut, 0) − max(min_mut − event, 0)``
            i.e. positive = past the most-cryptic-creating single, negative
            = below the least-active single (cryptic silencing).

    For annotated sites "more pathogenic" means *lower* probability; for
    cryptic sites it means *higher* probability.

    ``classification`` is the single label the rules assign (only one class can
    fire per site by construction). The class residuals are always populated
    so callers can inspect "near-miss" candidates for any mechanism.
    """
    position: int
    site_type: str        # 'donor' or 'acceptor'
    annotated: bool
    ref: float
    mut1: float
    mut2: float
    event: float
    expected: float       # mut1 + mut2 - ref
    residual: float       # event - expected  (signed; legacy/raw)
    classification: str   # one of CATEGORIES
    rescue_residual:           float
    cryptic_rescue_residual:   float
    deletion_synergy_residual: float
    cryptic_synergy_residual:  float
    residual_vs_expected:      float  # signed general residual vs additive null
    residual_vs_individuals:   float  # signed general residual vs the relevant single

    def as_dict(self) -> dict:
        d = asdict(self)
        d["position"] = int(d["position"])
        return d


def _general_epistasis_residuals(
    min_mut: float, max_mut: float, event: float,
    expected: float, annotated: bool,
) -> tuple[float, float]:
    """Return ``(residual_vs_expected, residual_vs_individuals)``.

    Both signed with ``+ve = synergy direction``, ``−ve = rescue direction``.
    See :class:`SiteResidual` for the formulas.
    """
    if annotated:
        # damaging direction = lower probability
        rve = expected - event                              # + ⇒ below additive (synergy)
        syn = max(min_mut - event, 0.0)                     # past worst (most damaged) single
        res = max(event - max_mut, 0.0)                     # past best  (least damaged) single
    else:
        # damaging direction = higher probability (cryptic activation)
        rve = event - expected                              # + ⇒ above additive (synergy)
        syn = max(event - max_mut, 0.0)                     # past worst (most active) single
        res = max(min_mut - event, 0.0)                     # past best  (least active) single
    return float(rve), float(syn - res)


# ─────────────────────────────────────────────────────────────────────────────
# Single-site classification (scalar). Used by the iterative pair path.
# ─────────────────────────────────────────────────────────────────────────────
def _classify_site(
    ref: float, mut1: float, mut2: float, event: float, residual: float,
    *,
    annotated: bool | None = None,
    residual_threshold: float = RESIDUAL_THRESHOLD,
    high_band:          float = HIGH_BAND,
    low_band:           float = LOW_BAND,
    near_wt:            float = NEAR_WT,
    # legacy kwargs (ignored) ───────────────────────────────────────
    total_effect_threshold:  float | None = None,
    single_effect_threshold: float | None = None,
    rescue_proximity:        float | None = None,
    activity_floor:          float | None = None,
) -> str:
    """Pair-case classification — returns one of :data:`CATEGORIES`.

    Note ``annotated`` is required for the WT-vs-annotation prerequisite —
    callers that don't pass it get ``"non-epistatic"`` (legacy fallback).
    """
    if annotated is None:
        return "non-epistatic"
    cls = _classify_site_full(
        ref, [mut1, mut2], event, bool(annotated),
        residual_threshold=residual_threshold,
        high_band=high_band, low_band=low_band, near_wt=near_wt,
    )[0]
    return cls


def _classify_site_multi(
    ref: float, muts: list[float], event: float, residual: float,
    *,
    annotated: bool | None = None,
    residual_threshold: float = RESIDUAL_THRESHOLD,
    high_band:          float = HIGH_BAND,
    low_band:           float = LOW_BAND,
    near_wt:            float = NEAR_WT,
    # legacy kwargs (ignored) ───────────────────────────────────────
    total_effect_threshold:  float | None = None,
    single_effect_threshold: float | None = None,
    rescue_proximity:        float | None = None,
    activity_floor:          float | None = None,
) -> str:
    if annotated is None:
        return "non-epistatic"
    cls = _classify_site_full(
        ref, list(muts), event, bool(annotated),
        residual_threshold=residual_threshold,
        high_band=high_band, low_band=low_band, near_wt=near_wt,
    )[0]
    return cls


def _classify_site_full(
    ref: float, muts: list[float], event: float, annotated: bool,
    *,
    residual_threshold: float = RESIDUAL_THRESHOLD,
    high_band:          float = HIGH_BAND,
    low_band:           float = LOW_BAND,
    near_wt:            float = NEAR_WT,
) -> tuple[str, float, float, float, float, float, float]:
    """Return ``(classification, rescue_res, cryptic_rescue_res,
    deletion_synergy_res, cryptic_synergy_res,
    residual_vs_expected, residual_vs_individuals)``.

    All four class-specific residuals are always computed and signed in
    *mechanism direction* (positive ⇒ that class would be supported by this
    site). ``classification`` selects which one fired. The two general
    residuals share the convention +ve = synergy, −ve = rescue — see
    :func:`_general_epistasis_residuals`.

    ``annotated`` must agree with the WT band, otherwise the site is
    non-epistatic regardless of mutation values:
      * annotated → ref ≥ high_band
      * not annotated → ref ≤ low_band
    """
    if not muts:
        return "non-epistatic", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    n = len(muts)
    min_mut = min(muts)
    max_mut = max(muts)
    expected = sum(muts) - (n - 1) * ref  # additive null on probability scale
    de = event - ref

    # mechanism-direction residuals, always computed (signed)
    rescue_res          = float(event - min_mut)        # +ve = joint above damaged single
    cryptic_rescue_res  = float(max_mut - event)        # +ve = joint below cryptic creator
    deletion_synergy_res = float(expected - event)      # +ve = joint below additive
    cryptic_synergy_res  = float(event - expected)      # +ve = joint above additive
    rve, rvi             = _general_epistasis_residuals(
        min_mut, max_mut, event, expected, annotated,
    )

    # ── WT-vs-annotation prerequisite (the noise filter) ────────────
    if annotated:
        if ref < high_band:
            return ("non-epistatic", rescue_res, cryptic_rescue_res,
                    deletion_synergy_res, cryptic_synergy_res, rve, rvi)
        # ── deletion_synergy
        if (
            min_mut >= high_band
            and (ref - event) >= residual_threshold
            and deletion_synergy_res >= residual_threshold
        ):
            return ("deletion_synergy", rescue_res, cryptic_rescue_res,
                    deletion_synergy_res, cryptic_synergy_res, rve, rvi)
        # ── rescue
        if (
            min_mut <= (ref - high_band)
            and abs(de) <= near_wt
            and rescue_res >= residual_threshold
        ):
            return ("rescue", rescue_res, cryptic_rescue_res,
                    deletion_synergy_res, cryptic_synergy_res, rve, rvi)
        return ("non-epistatic", rescue_res, cryptic_rescue_res,
                deletion_synergy_res, cryptic_synergy_res, rve, rvi)

    # ── unannotated branch ──────────────────────────────────────────
    if ref > low_band:
        return ("non-epistatic", rescue_res, cryptic_rescue_res,
                deletion_synergy_res, cryptic_synergy_res, rve, rvi)
    # ── cryptic_synergy
    if (
        max_mut <= low_band
        and event >= high_band
        and cryptic_synergy_res >= residual_threshold
    ):
        return ("cryptic_synergy", rescue_res, cryptic_rescue_res,
                deletion_synergy_res, cryptic_synergy_res, rve, rvi)
    # ── cryptic_rescue
    if (
        max_mut >= high_band
        and event <= low_band
        and cryptic_rescue_res >= residual_threshold
    ):
        return ("cryptic_rescue", rescue_res, cryptic_rescue_res,
                deletion_synergy_res, cryptic_synergy_res, rve, rvi)
    return ("non-epistatic", rescue_res, cryptic_rescue_res,
            deletion_synergy_res, cryptic_synergy_res, rve, rvi)


# ─────────────────────────────────────────────────────────────────────────────
# Frame-level entry points (pair + N-variant), vectorized.
# ─────────────────────────────────────────────────────────────────────────────
def compute_site_residuals(
    site_table: pd.DataFrame,
    *,
    threshold:          float = RESIDUAL_THRESHOLD,
    high_band:          float = HIGH_BAND,
    low_band:           float = LOW_BAND,
    near_wt:            float = NEAR_WT,
    activity_min:       float = 0.0,
    # legacy kwargs (silently accepted, ignored) ─────────────────────
    total_effect_threshold:  float | None = None,
    single_effect_threshold: float | None = None,
    rescue_proximity:        float | None = None,
    activity_floor:          float | None = None,
) -> pd.DataFrame:
    """Per-site (ref, mut1, mut2, event, expected, residual, classification).

    See the module docstring for the 4-class definitions.
    """
    return compute_site_residuals_multi(
        site_table, n_variants=2,
        threshold=threshold, high_band=high_band, low_band=low_band,
        near_wt=near_wt, activity_min=activity_min,
    )


def compute_site_residuals_multi(
    site_table: pd.DataFrame,
    n_variants: int,
    *,
    threshold:          float = RESIDUAL_THRESHOLD,
    high_band:          float = HIGH_BAND,
    low_band:           float = LOW_BAND,
    near_wt:            float = NEAR_WT,
    activity_min:       float = 0.0,
    # legacy kwargs (silently accepted, ignored) ─────────────────────
    total_effect_threshold:  float | None = None,
    single_effect_threshold: float | None = None,
    rescue_proximity:        float | None = None,
    activity_floor:          float | None = None,
) -> pd.DataFrame:
    """N-variant per-site classification.

    Required contexts: ``ref``, ``mut1``, …, ``mut{N}``, ``event``. The
    classification generalises by collapsing the singles to their min/max as
    the rules dictate (rescue: ``min`` of singles for high-band, ``max`` for
    low-band; synergies: requires *all* singles in the band).
    """
    import numpy as _np

    wide = site_table_wide(site_table)
    required = {"ref", "event"} | {f"mut{i}" for i in range(1, n_variants + 1)}
    missing = required - set(wide.columns)
    if missing:
        raise ValueError(
            f"site_table missing contexts for {n_variants}-variant analysis: {sorted(missing)}"
        )

    ref   = wide["ref"].to_numpy(dtype=float, copy=False)
    event = wide["event"].to_numpy(dtype=float, copy=False)
    muts  = _np.column_stack([
        wide[f"mut{i}"].to_numpy(dtype=float, copy=False)
        for i in range(1, n_variants + 1)
    ])  # (N_sites, n_variants)

    # activity floor: drop sites that are totally silent across every context
    abs_all = _np.maximum.reduce(
        [_np.abs(ref), _np.abs(event)] + [_np.abs(muts[:, i]) for i in range(n_variants)]
    )
    keep = abs_all >= activity_min
    n_kept = int(keep.sum())
    if n_kept == 0:
        return pd.DataFrame(columns=[
            "position", "site_type", "annotated", "ref",
            "mut1", "mut2", "event", "expected", "residual",
            "classification",
            "rescue_residual", "cryptic_rescue_residual",
            "deletion_synergy_residual", "cryptic_synergy_residual",
        ])

    ref_k   = ref[keep]
    event_k = event[keep]
    muts_k  = muts[keep]
    annotated_k = wide["annotated"].to_numpy()[keep].astype(bool)

    min_mut = muts_k.min(axis=1)
    max_mut = muts_k.max(axis=1)
    expected = muts_k.sum(axis=1) - (n_variants - 1) * ref_k
    residual = event_k - expected
    de       = event_k - ref_k

    # ── WT-vs-annotation prerequisite: only sites where the engine agrees
    #    with the annotation are eligible for any class.
    ann_ok  = annotated_k  & (ref_k >= high_band)   # annotated branch
    cryp_ok = ~annotated_k & (ref_k <= low_band)    # cryptic branch

    # ── deletion_synergy (annotated branch)
    del_syn = ann_ok & (
        (min_mut >= high_band)
        & ((ref_k    - event_k) >= threshold)
        & ((expected - event_k) >= threshold)
    )

    # ── rescue (annotated branch, only when del_syn doesn't claim the site)
    rescue = ann_ok & (
        (min_mut <= (ref_k - high_band))
        & (_np.abs(de) <= near_wt)
        & ((event_k - min_mut) >= threshold)
    ) & ~del_syn

    # ── cryptic_synergy (unannotated branch)
    cryp_syn = cryp_ok & (
        (max_mut <= low_band)
        & (event_k >= high_band)
        & ((event_k - expected) >= threshold)
    )

    # ── cryptic_rescue (unannotated branch)
    cryp_rescue = cryp_ok & (
        (max_mut >= high_band)
        & (event_k <= low_band)
        & ((max_mut - event_k) >= threshold)
    ) & ~cryp_syn

    classification = _np.full(n_kept, "non-epistatic", dtype=object)
    classification[del_syn]     = "deletion_synergy"
    classification[cryp_syn]    = "cryptic_synergy"
    classification[rescue]      = "rescue"
    classification[cryp_rescue] = "cryptic_rescue"

    # All 4 mechanism-direction residuals are *always* computed at every
    # site (signed; positive = the named mechanism is supported). The
    # ``classification`` column selects which one fired.
    rescue_residual            = event_k   - min_mut     # joint above damaged single
    cryptic_rescue_residual    = max_mut   - event_k     # joint below cryptic creator
    deletion_synergy_residual  = expected  - event_k     # joint below additive
    cryptic_synergy_residual   = event_k   - expected    # joint above additive

    # Two general signed residuals, both with +ve = synergy, -ve = rescue.
    zero = _np.zeros_like(event_k)
    # vs additive expected: classical residual (sign-flipped for cryptic)
    rve = _np.where(annotated_k, expected - event_k, event_k - expected)
    # vs individual mutations: past worst (synergy) or best (rescue) single, 0 between.
    # Annotated: worst=min, best=max; Cryptic: worst=max, best=min.
    syn_dist = _np.where(
        annotated_k,
        _np.maximum(min_mut - event_k, zero),    # event below worst-damaged single
        _np.maximum(event_k - max_mut, zero),    # event above worst-cryptic single
    )
    res_dist = _np.where(
        annotated_k,
        _np.maximum(event_k - max_mut, zero),    # event above best-preserved single
        _np.maximum(min_mut - event_k, zero),    # event below least-active single
    )
    rvi = syn_dist - res_dist

    out = pd.DataFrame({
        "position":       wide["position"].to_numpy()[keep].astype(int),
        "site_type":      wide["site_type"].to_numpy()[keep].astype(str),
        "annotated":      wide["annotated"].to_numpy()[keep].astype(bool),
        "ref":            ref_k,
        "mut1":           muts_k[:, 0] if n_variants >= 1 else 0.0,
        "mut2":           muts_k[:, 1] if n_variants >= 2 else 0.0,
        "event":          event_k,
        "expected":       expected,
        "residual":       residual,
        "classification": classification,
        "rescue_residual":           rescue_residual,
        "cryptic_rescue_residual":   cryptic_rescue_residual,
        "deletion_synergy_residual": deletion_synergy_residual,
        "cryptic_synergy_residual":  cryptic_synergy_residual,
        "residual_vs_expected":      rve,
        "residual_vs_individuals":   rvi,
    })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Pair-level aggregation + summary.
# ─────────────────────────────────────────────────────────────────────────────
_CLASS_RESIDUAL_COL = {
    "rescue":           "rescue_residual",
    "cryptic_rescue":   "cryptic_rescue_residual",
    "deletion_synergy": "deletion_synergy_residual",
    "cryptic_synergy":  "cryptic_synergy_residual",
}


def classify_pair(site_residuals: pd.DataFrame) -> str:
    """Aggregate per-site classifications into one pair-level label.

    The label is the class of the site with the largest *class-specific*
    residual (the column matching that site's classification). Ties are
    broken by :data:`CATEGORIES` priority (synergies before rescues).
    """
    if site_residuals.empty or "classification" not in site_residuals.columns:
        return "non-epistatic"
    df = site_residuals[site_residuals.classification.isin(EPISTATIC)]
    if df.empty:
        return "non-epistatic"

    # pick the class-specific residual for each row
    def _mag(row) -> float:
        col = _CLASS_RESIDUAL_COL.get(row.classification)
        return float(row[col]) if col and col in row.index else 0.0

    mags = df.apply(_mag, axis=1)
    best_idx = mags.idxmax()
    tied = df[mags == mags.loc[best_idx]]
    if len(tied) > 1:
        tied = tied.assign(_pri=tied.classification.map(_PRIORITY))
        best = tied.sort_values("_pri").iloc[0]
    else:
        best = df.loc[best_idx]
    return str(best.classification)


def _max_class_residual(site_residuals: pd.DataFrame, cls: str) -> float:
    """Max of the class-specific residual over sites that *fired* this class."""
    col = _CLASS_RESIDUAL_COL[cls]
    if col not in site_residuals.columns:
        return 0.0
    fired = site_residuals[site_residuals.classification == cls]
    return float(fired[col].max()) if not fired.empty else 0.0


def summarize_residuals(site_residuals: pd.DataFrame) -> dict:
    """One-line numerical summary of a pair's residual landscape.

    Provides per-class site counts plus the max class-specific residual for
    each of the four mechanism classes (over sites that *fired* that class).
    """
    if site_residuals.empty:
        return {
            "n_sites": 0,
            "n_del_syn": 0, "n_cryp_syn": 0, "n_rescue": 0, "n_cryp_rescue": 0,
            "max_rescue_residual":           0.0,
            "max_cryptic_rescue_residual":   0.0,
            "max_deletion_synergy_residual": 0.0,
            "max_cryptic_synergy_residual":  0.0,
            "max_abs_residual":    0.0,
            "max_abs_event_delta": 0.0,
            "pair_classification": "non-epistatic",
        }
    cls = site_residuals.classification
    delta_event = (site_residuals.event - site_residuals.ref).abs()

    return {
        "n_sites":              int(len(site_residuals)),
        "n_del_syn":            int((cls == "deletion_synergy").sum()),
        "n_cryp_syn":           int((cls == "cryptic_synergy").sum()),
        "n_rescue":             int((cls == "rescue").sum()),
        "n_cryp_rescue":        int((cls == "cryptic_rescue").sum()),
        "max_rescue_residual":           _max_class_residual(site_residuals, "rescue"),
        "max_cryptic_rescue_residual":   _max_class_residual(site_residuals, "cryptic_rescue"),
        "max_deletion_synergy_residual": _max_class_residual(site_residuals, "deletion_synergy"),
        "max_cryptic_synergy_residual":  _max_class_residual(site_residuals, "cryptic_synergy"),
        "max_abs_residual":     float(site_residuals.residual.abs().max()),
        "max_abs_event_delta":  float(delta_event.max()),
        "pair_classification":  classify_pair(site_residuals),
    }
