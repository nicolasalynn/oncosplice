"""
Splicing-epistasis math: per-site expected/observed/residual + classification.

For each splice site, given probabilities in four contexts (ref, mut1, mut2,
event), we define::

    delta1   = mut1   - ref
    delta2   = mut2   - ref
    delta_event = event - ref
    expected = delta1 + delta2     (additive null)
    residual = delta_event - expected

Five-category classification (oncosplice ≥ 3.2.0)
-------------------------------------------------

A splice site is classified into exactly one of five buckets based on the
relationship between ``delta_event`` and ``expected``:

- **synergistic** — joint effect strictly *larger* than additive
  (``|residual| ≥ residual_threshold`` AND ``|delta_event| > |expected|``).

- **rescue** — at least one single variant disrupts splicing on its own
  (``max(|delta1|, |delta2|) ≥ single_effect_threshold``) AND the joint
  variant restores splicing close to wild-type
  (``|delta_event| ≤ rescue_proximity``). Captures the case where one
  variant cancels the splicing impact of the other, even if the *residual*
  itself is modest.

- **compounding** — residual is small (additive expectation holds) BUT
  the **joint effect itself is large** (``|delta_event| ≥ total_effect_threshold``)
  and **both** singles contribute substantially in the same direction.
  Captures cases like ``mut1 +0.3, mut2 +0.3, joint +0.6`` where there's no
  super-additivity but the biological joint impact is undeniable.

- **antagonistic** — joint effect *smaller* than additive
  (``|residual| ≥ residual_threshold`` AND ``|delta_event| < |expected|``)
  but does not meet the rescue criterion.

- **non-epistatic** — none of the above.

Pair-level aggregation
----------------------

A pair gets one label by descending priority across all its sites:

    synergistic  >  rescue  >  compounding  >  antagonistic  >  non-epistatic

This mirrors the comut_tracking manuscript's "any-site-wins" rule but
extended to the new categories so a single rescue site doesn't get hidden
behind a compounding one elsewhere in the gene.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List

import pandas as pd

from .splicing import site_table_wide

DEFAULT_RESIDUAL_THRESHOLD       = 0.25  # |residual| for syn/ant
DEFAULT_TOTAL_EFFECT_THRESHOLD   = 0.50  # |delta_event| for compounding
DEFAULT_SINGLE_EFFECT_THRESHOLD  = 0.15  # min |delta_i| for compounding/rescue trigger
DEFAULT_RESCUE_PROXIMITY         = 0.20  # |delta_event| ≤ this counts as "near WT"
DEFAULT_ACTIVITY_FLOOR           = 0.10  # max-context floor for any classification

CATEGORIES = ("synergistic", "rescue", "compounding", "non-epistatic")
_PRIORITY  = {c: i for i, c in enumerate(CATEGORIES)}  # lower index = higher priority


@dataclass(frozen=True)
class SiteResidual:
    """Per-splice-site epistasis breakdown."""
    position: int
    site_type: str        # 'donor' or 'acceptor'
    annotated: bool
    ref: float
    mut1: float
    mut2: float
    event: float
    expected: float       # ref + delta1 + delta2  ≡  mut1 + mut2 - ref
    residual: float       # event - expected
    classification: str   # one of CATEGORIES

    def as_dict(self) -> dict:
        d = asdict(self)
        d["position"] = int(d["position"])
        return d


def _sign(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


def _classify_site(
    ref: float, mut1: float, mut2: float, event: float, residual: float,
    *,
    residual_threshold:      float = DEFAULT_RESIDUAL_THRESHOLD,
    total_effect_threshold:  float = DEFAULT_TOTAL_EFFECT_THRESHOLD,
    single_effect_threshold: float = DEFAULT_SINGLE_EFFECT_THRESHOLD,
    rescue_proximity:        float = DEFAULT_RESCUE_PROXIMITY,
    activity_floor:          float = DEFAULT_ACTIVITY_FLOOR,
) -> str:
    """Pair-case shim around :func:`_classify_site_multi` — same three-case rules."""
    return _classify_site_multi(
        ref, [mut1, mut2], event, residual,
        residual_threshold=residual_threshold,
        total_effect_threshold=total_effect_threshold,
        single_effect_threshold=single_effect_threshold,
        rescue_proximity=rescue_proximity,
        activity_floor=activity_floor,
    )


def _classify_site_multi(
    ref: float, muts: list[float], event: float, residual: float,
    *,
    residual_threshold:      float = DEFAULT_RESIDUAL_THRESHOLD,
    total_effect_threshold:  float = DEFAULT_TOTAL_EFFECT_THRESHOLD,
    single_effect_threshold: float = DEFAULT_SINGLE_EFFECT_THRESHOLD,
    rescue_proximity:        float = DEFAULT_RESCUE_PROXIMITY,
    activity_floor:          float = DEFAULT_ACTIVITY_FLOOR,
) -> str:
    """Three-case site classification, anchored on (d_singles, d_event).

    Drops the additive-residual baseline because probabilities are bounded in
    [0, 1] — additive expectations saturate and produce spurious antagonistic
    / compounding calls. All decisions compare the joint Δ to the **worst**
    single Δ (the single with the largest |Δ|).

    Rules:

    * **rescue** — worst single makes a real change (|worst| ≥ 0.30) AND the
      joint returns close to WT (|de| ≤ 0.20) AND the rescue is real (joint
      is at least 0.15 closer to WT than worst).
    * **synergistic** — joint goes further than the worst single in the same
      direction by more than 25% of the worst single's magnitude, AND the
      joint is itself substantial (|de| ≥ 0.25).
    * **compounding** — same as synergistic but the extra push is ≤25% of
      the worst single (slight additional effect).
    * **non-epistatic** — everything else. Includes saturation (joint matches
      worst), redundant disruption (both singles already destroy the site),
      and weak signals.

    The function still accepts ``residual`` and the historical threshold
    kwargs for API compatibility — they are ignored.
    """
    de = event - ref
    abs_de = abs(de)
    deltas = [m - ref for m in muts]
    abs_deltas = [abs(d) for d in deltas]

    if not abs_deltas:
        return "non-epistatic"
    worst_idx     = max(range(len(deltas)), key=lambda i: abs_deltas[i])
    worst_signed  = deltas[worst_idx]
    worst_abs     = abs_deltas[worst_idx]
    min_abs       = min(abs_deltas)
    expected_delta = sum(deltas)
    abs_expected   = abs(expected_delta)

    # ── 1. Rescue: substantial single, joint near WT, real reduction.
    # Same-side check only matters when joint is appreciably away from WT —
    # tiny floating-point overshoots past WT (e.g. de=0.0008) shouldn't disqualify.
    near_wt = abs_de <= 0.05
    same_side_ok = near_wt or (worst_signed * de >= 0)
    if (worst_abs >= 0.30
            and abs_de   <= 0.20
            and (worst_abs - abs_de) >= 0.15
            and same_side_ok):
        return "rescue"

    # ── 2. Flip-synergy: worst goes one way, joint goes the OTHER way,
    #      AND joint differs meaningfully from EVERY single. Without the
    #      "differs from all" check, this fires whenever the opposing
    #      single dominates — which is just dominance, not epistasis.
    flip_min_diff = min(abs(event - m) for m in muts) if muts else 0.0
    if (worst_signed * de < 0
            and worst_abs >= 0.20
            and abs_de    >= 0.15
            and flip_min_diff >= 0.15):
        return "synergistic"

    # ── 3. Emergent at edge: ref pegged near 0 or 1, singles barely move,
    #      joint moves to a meaningfully different state (logit-scale meaningful).
    ref_at_edge = (ref <= 0.10) or (ref >= 0.90)
    if ref_at_edge and worst_abs < 0.10 and abs_de >= 0.10:
        return "synergistic"

    # Remaining buckets all require: joint substantial, in same direction as worst,
    # past the worst single, AND past the additive expectation (residual > 0 in
    # joint's direction — i.e., joint actually went further than singles predict).
    same_direction = (worst_signed == 0 and abs_de > 0) or (worst_signed * de > 0)
    if not (same_direction
            and abs_de >= 0.25
            and abs_de > worst_abs
            and abs_de > abs_expected):
        return "non-epistatic"

    # ``residual`` here = how far the joint went BEYOND the additive expectation,
    # in the joint's direction. This is the user's "residual" — the amount by
    # which the joint exceeds what the singles together would predict.
    residual_over_additive = abs_de - abs_expected
    threshold = 0.25 * worst_abs

    # ── 4. Synergistic: residual is substantial relative to the worst single.
    if residual_over_additive > threshold:
        return "synergistic"

    # ── 5. Compounding: residual is positive but small AND both mutations
    #      contribute substantially (min |d| ≥ 0.20). Without both contributing,
    #      a small residual is just dominance + noise.
    if min_abs >= 0.20:
        return "compounding"

    return "non-epistatic"


def compute_site_residuals(
    site_table: pd.DataFrame,
    *,
    threshold:               float = DEFAULT_RESIDUAL_THRESHOLD,
    total_effect_threshold:  float = DEFAULT_TOTAL_EFFECT_THRESHOLD,
    single_effect_threshold: float = DEFAULT_SINGLE_EFFECT_THRESHOLD,
    rescue_proximity:        float = DEFAULT_RESCUE_PROXIMITY,
    activity_floor:          float = DEFAULT_ACTIVITY_FLOOR,
    activity_min:            float = 0.0,
) -> pd.DataFrame:
    """Per-site (ref, mut1, mut2, event, expected, residual, classification) DataFrame.

    Five-category classification (see module docstring). ``threshold`` is the
    legacy name for ``residual_threshold`` — kept for backwards compat.
    """
    wide = site_table_wide(site_table)
    required = {"ref", "mut1", "mut2", "event"}
    missing = required - set(wide.columns)
    if missing:
        raise ValueError(
            f"site_table is missing required contexts: {sorted(missing)}. "
            "Did you run predict_splicing with individual_mutations=True?"
        )

    rows: List[SiteResidual] = []
    for _, row in wide.iterrows():
        ref = float(row.get("ref", 0.0) or 0.0)
        m1  = float(row.get("mut1", 0.0) or 0.0)
        m2  = float(row.get("mut2", 0.0) or 0.0)
        ev  = float(row.get("event", 0.0) or 0.0)
        if max(abs(ref), abs(m1), abs(m2), abs(ev)) < activity_min:
            continue
        expected = m1 + m2 - ref
        residual = ev - expected
        cls = _classify_site(
            ref, m1, m2, ev, residual,
            residual_threshold=threshold,
            total_effect_threshold=total_effect_threshold,
            single_effect_threshold=single_effect_threshold,
            rescue_proximity=rescue_proximity,
            activity_floor=activity_floor,
        )
        rows.append(SiteResidual(
            position=int(row["position"]),
            site_type=str(row["site_type"]),
            annotated=bool(row["annotated"]),
            ref=ref, mut1=m1, mut2=m2, event=ev,
            expected=expected, residual=residual,
            classification=cls,
        ))
    if not rows:
        return pd.DataFrame(columns=[
            "position", "site_type", "annotated",
            "ref", "mut1", "mut2", "event",
            "expected", "residual", "classification",
        ])
    return pd.DataFrame([r.as_dict() for r in rows])


def compute_site_residuals_multi(
    site_table: pd.DataFrame,
    n_variants: int,
    *,
    threshold:               float = DEFAULT_RESIDUAL_THRESHOLD,
    total_effect_threshold:  float = DEFAULT_TOTAL_EFFECT_THRESHOLD,
    single_effect_threshold: float = DEFAULT_SINGLE_EFFECT_THRESHOLD,
    rescue_proximity:        float = DEFAULT_RESCUE_PROXIMITY,
    activity_floor:          float = DEFAULT_ACTIVITY_FLOOR,
    activity_min:            float = 0.0,
) -> pd.DataFrame:
    """N-variant generalisation of :func:`compute_site_residuals`.

    Required contexts: ``ref``, ``mut1``, …, ``mut{N}``, ``event``. The
    additive expectation at each site is ``sum(mut_i) - (N-1)·ref`` and the
    residual is ``event - expected``. Same five-category classification as
    the pair case, generalised to any N≥2.
    """
    wide = site_table_wide(site_table)
    required = {"ref", "event"} | {f"mut{i}" for i in range(1, n_variants + 1)}
    missing = required - set(wide.columns)
    if missing:
        raise ValueError(f"site_table missing contexts for {n_variants}-variant analysis: {sorted(missing)}")

    # Vectorized path: ~50× faster than the iterrows loop on real gene scans.
    # All computation is numpy arrays + boolean masks; the priority hierarchy
    # is preserved via order-of-assignment.
    import numpy as _np

    ref   = wide["ref"].to_numpy(dtype=float, copy=False)
    event = wide["event"].to_numpy(dtype=float, copy=False)
    muts  = _np.column_stack([
        wide[f"mut{i}"].to_numpy(dtype=float, copy=False) for i in range(1, n_variants + 1)
    ])  # (N_sites, n_variants)

    # ---- activity floor (applies the activity_min filter from the API) ----
    abs_all = _np.maximum.reduce(
        [_np.abs(ref), _np.abs(event)] + [_np.abs(muts[:, i]) for i in range(n_variants)]
    )
    keep = abs_all >= activity_min
    if not keep.any():
        return pd.DataFrame(columns=[
            "position", "site_type", "annotated", "ref",
            "mut1", "mut2", "event", "expected", "residual", "classification",
        ])

    ref_k = ref[keep]; event_k = event[keep]; muts_k = muts[keep]
    deltas = muts_k - ref_k[:, None]                                  # (k, n)
    # `expected`/`residual` retained for the output frame (consumers downstream
    # still want them as metadata).
    expected_delta = deltas.sum(axis=1)
    expected = ref_k + expected_delta
    residual = event_k - expected

    de        = event_k - ref_k
    abs_de    = _np.abs(de)
    abs_d     = _np.abs(deltas)
    abs_expected = _np.abs(expected_delta)

    # Worst single = the single with the largest |delta|.
    worst_idx    = abs_d.argmax(axis=1)
    rows         = _np.arange(deltas.shape[0])
    worst_signed = deltas[rows, worst_idx]
    worst_abs    = abs_d   [rows, worst_idx]
    min_abs      = abs_d.min(axis=1)

    # ── rescue (same-side check relaxed when joint is essentially at WT)
    same_side_ok = (abs_de <= 0.05) | (worst_signed * de >= 0)
    rescue = (
        (worst_abs >= 0.30)
        & (abs_de <= 0.20)
        & ((worst_abs - abs_de) >= 0.15)
        & same_side_ok
    )

    # ── flip-synergy (also requires joint to differ from EVERY single,
    #    so dominance of the opposing single isn't mis-labelled as flip)
    joint_minus_muts = event_k[:, None] - muts_k                 # (k, n_variants)
    flip_min_diff    = _np.abs(joint_minus_muts).min(axis=1)
    flip_syn = (
        (worst_signed * de < 0)
        & (worst_abs >= 0.20)
        & (abs_de >= 0.15)
        & (flip_min_diff >= 0.15)
        & ~rescue
    )

    # ── emergent at edge
    ref_at_edge = (ref_k <= 0.10) | (ref_k >= 0.90)
    emergent_syn = ref_at_edge & (worst_abs < 0.10) & (abs_de >= 0.10) & ~rescue & ~flip_syn

    # ── shared guard: joint substantial, same direction, past worst AND past additive
    same_dir = ((worst_signed == 0) & (abs_de > 0)) | (worst_signed * de > 0)
    main_ok  = (
        same_dir
        & (abs_de >= 0.25)
        & (abs_de > worst_abs)
        & (abs_de > abs_expected)
        & ~rescue & ~flip_syn & ~emergent_syn
    )

    # ``residual`` = how far joint went past the additive expectation.
    residual_over_additive = abs_de - abs_expected
    threshold = 0.25 * worst_abs

    # ── synergistic: residual substantial relative to worst single
    syn_main = main_ok & (residual_over_additive > threshold)

    # ── compounding: residual positive-but-small, BOTH contribute
    comp_main = main_ok & ~syn_main & (min_abs >= 0.20)

    syn  = syn_main | emergent_syn | flip_syn
    comp = comp_main

    classification = _np.full(keep.sum(), "non-epistatic", dtype=object)
    classification[comp]   = "compounding"
    classification[rescue] = "rescue"
    classification[syn]    = "synergistic"

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
    })
    return out


def classify_pair(site_residuals: pd.DataFrame) -> str:
    """Aggregate site-level classifications into one pair-level call.

    Priority hierarchy (highest first): synergistic > rescue > compounding >
    antagonistic > non-epistatic. The pair gets the highest-priority label
    that appears at any site.
    """
    if site_residuals.empty or "classification" not in site_residuals.columns:
        return "non-epistatic"
    present = set(site_residuals["classification"].unique())
    for cat in CATEGORIES:
        if cat in present:
            return cat
    return "non-epistatic"


def summarize_residuals(site_residuals: pd.DataFrame) -> dict:
    """One-line numerical summary of a pair's residual landscape."""
    if site_residuals.empty:
        return {
            "n_sites": 0,
            "n_syn": 0, "n_rescue": 0, "n_compound": 0, "n_ant": 0,
            "max_abs_residual": 0.0,
            "max_abs_event_delta": 0.0,
            "max_synergy_residual":     0.0,
            "max_antagonism_residual":  0.0,
            "pair_classification": "non-epistatic",
        }
    syn      = site_residuals[site_residuals.classification == "synergistic"]
    rescue   = site_residuals[site_residuals.classification == "rescue"]
    compound = site_residuals[site_residuals.classification == "compounding"]
    ant      = site_residuals[site_residuals.classification == "antagonistic"]

    def _max_abs(series: pd.Series) -> float:
        return float(series.abs().max()) if not series.empty else 0.0

    delta_event = (site_residuals.event - site_residuals.ref).abs()

    return {
        "n_sites":     int(len(site_residuals)),
        "n_syn":       int(len(syn)),
        "n_rescue":    int(len(rescue)),
        "n_compound":  int(len(compound)),
        "n_ant":       int(len(ant)),
        "max_abs_residual":         float(site_residuals.residual.abs().max()),
        "max_abs_event_delta":      float(delta_event.max()),
        "max_synergy_residual":     _max_abs(syn.residual),
        "max_antagonism_residual":  _max_abs(ant.residual),
        "pair_classification":      classify_pair(site_residuals),
    }
