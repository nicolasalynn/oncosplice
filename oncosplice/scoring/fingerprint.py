"""Coarse fingerprints + stable hashes for splicing outcomes.

The idea: group events (single or double variants) by their **gross
splicing impact pattern**, not their exact numerical predictions. Two
events with the same fingerprint produced the same kind of splicing
change (e.g. "lost 1 annotated donor, gained 1 cryptic acceptor")
regardless of where in the gene the changes happened, or whether one
prediction was 0.85 and the other 0.87.

Fingerprint string format
-------------------------

Single variants::

    "D-2.A+1"        2 lost donors, 1 gained cryptic acceptor
    "_"              no significant change

Double variants append a mechanism tag::

    "D-1.A+1|rescue;res:1,csyn:1" outcome + 1 rescue site + 1 cryptic-synergy site
    "_|non-epistatic"             no splicing change, no epistasis

Use ``splicing_outcome_hash(result)`` to get a stable 16-char hex string
suitable as a dict key for population-scale grouping.
"""
from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..results import DoubleVariantResult, SingleVariantResult

DEFAULT_DELTA   = 0.25   # |Δprob| threshold for "lost" / "gained"
DEFAULT_ACTIVE  = 0.10   # max-context activity floor to count a site at all


# ──────────────────────────────────────────────────────────────────────────
# Single variant
# ──────────────────────────────────────────────────────────────────────────

def _outcome_counts_single(result: "SingleVariantResult", *,
                            delta: float = DEFAULT_DELTA) -> dict:
    """Return {n_lost_donors, n_lost_acceptors, n_gained_donors, n_gained_acceptors}
    derived from a SingleVariantResult.missplicing dict.
    """
    m = result.missplicing
    def _count(bucket):
        return sum(1 for v in bucket.values()
                   if abs(v.get("delta", 0.0)) >= delta)
    return {
        "n_lost_donors":     _count(m.missed_donors),
        "n_lost_acceptors":  _count(m.missed_acceptors),
        "n_gained_donors":   _count(m.discovered_donors),
        "n_gained_acceptors": _count(m.discovered_acceptors),
    }


# ──────────────────────────────────────────────────────────────────────────
# Double variant
# ──────────────────────────────────────────────────────────────────────────

def _outcome_counts_double(result: "DoubleVariantResult", *,
                            delta: float = DEFAULT_DELTA,
                            active_floor: float = DEFAULT_ACTIVE) -> dict:
    """Same shape as the single version, computed from
    DoubleVariantResult.site_residuals (the joint context vs reference).
    """
    sr = result.site_residuals
    if sr.empty:
        return {"n_lost_donors": 0, "n_lost_acceptors": 0,
                "n_gained_donors": 0, "n_gained_acceptors": 0}
    delta_event = (sr.event - sr.ref)
    activity = sr[["ref", "mut1", "mut2", "event"]].abs().max(axis=1)
    flagged = (delta_event.abs() >= delta) & (activity >= active_floor)
    sub = sr[flagged]
    losses_donor    = ((sub.site_type == "donor")    & sub.annotated & (delta_event[flagged] < 0)).sum()
    losses_acceptor = ((sub.site_type == "acceptor") & sub.annotated & (delta_event[flagged] < 0)).sum()
    gains_donor     = ((sub.site_type == "donor")    & ~sub.annotated & (delta_event[flagged] > 0)).sum()
    gains_acceptor  = ((sub.site_type == "acceptor") & ~sub.annotated & (delta_event[flagged] > 0)).sum()
    return {
        "n_lost_donors":     int(losses_donor),
        "n_lost_acceptors":  int(losses_acceptor),
        "n_gained_donors":   int(gains_donor),
        "n_gained_acceptors": int(gains_acceptor),
    }


def _format_outcome(counts: dict) -> str:
    """Compact outcome string from the count dict. ``"_"`` if all zero."""
    parts = []
    if counts["n_lost_donors"]:     parts.append(f"D-{counts['n_lost_donors']}")
    if counts["n_lost_acceptors"]:  parts.append(f"A-{counts['n_lost_acceptors']}")
    if counts["n_gained_donors"]:   parts.append(f"D+{counts['n_gained_donors']}")
    if counts["n_gained_acceptors"]: parts.append(f"A+{counts['n_gained_acceptors']}")
    return ".".join(parts) if parts else "_"


# ──────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────

def splicing_outcome_fingerprint(
    result,
    *,
    include_mechanism: bool = True,
    delta: float = DEFAULT_DELTA,
    active_floor: float = DEFAULT_ACTIVE,
) -> str:
    """Compact human-readable fingerprint of a splicing outcome.

    Parameters
    ----------
    result
        :class:`~oncosplice.SingleVariantResult` or
        :class:`~oncosplice.DoubleVariantResult` (or
        :class:`~oncosplice.MultiVariantResult` — also supported).
    include_mechanism
        For double / multi results, append the pair-level classification
        + per-class site counts (e.g. ``"|rescue;res:1,csyn:1"``).
    delta
        |Δprob| threshold for counting a site as lost/gained (default
        0.25, matches the residual threshold).
    active_floor
        Sites whose max-context probability is below this are ignored.

    Returns
    -------
    str
        Format ``"D-2.A+1"`` (singles) or ``"D-2.A+1|rescue;res:1,csyn:1"``
        (doubles). Use ``"_"`` for "no significant change".

    Examples
    --------
    >>> # Group double variants by gross splicing impact
    >>> df["fp"] = df.apply(lambda r: splicing_outcome_fingerprint(r.result), axis=1)
    >>> df.groupby("fp").size().sort_values(ascending=False).head()
    """
    from ..results import DoubleVariantResult, MultiVariantResult, SingleVariantResult

    if isinstance(result, SingleVariantResult):
        counts = _outcome_counts_single(result, delta=delta)
        return _format_outcome(counts)

    if isinstance(result, (DoubleVariantResult, MultiVariantResult)):
        counts = _outcome_counts_double(result, delta=delta, active_floor=active_floor)
        outcome = _format_outcome(counts)
        if not include_mechanism:
            return outcome
        summary = getattr(result, "epistasis_summary", None) or {}
        cls = getattr(result, "pair_classification", "non-epistatic")
        # 4-class taxonomy (matches summarize_residuals keys).
        mech_parts = []
        if summary.get("n_del_syn"):     mech_parts.append(f"dsyn:{summary['n_del_syn']}")
        if summary.get("n_cryp_syn"):    mech_parts.append(f"csyn:{summary['n_cryp_syn']}")
        if summary.get("n_rescue"):      mech_parts.append(f"res:{summary['n_rescue']}")
        if summary.get("n_cryp_rescue"): mech_parts.append(f"cres:{summary['n_cryp_rescue']}")
        mech_str = (cls + (";" + ",".join(mech_parts) if mech_parts else ""))
        return f"{outcome}|{mech_str}"

    raise TypeError(f"unsupported result type: {type(result).__name__}")


def splicing_outcome_hash(result, **kwargs) -> str:
    """16-char hex stable hash of :func:`splicing_outcome_fingerprint`.

    Suitable as a dict / DataFrame group key for population-scale grouping.
    """
    fp = splicing_outcome_fingerprint(result, **kwargs)
    return hashlib.md5(fp.encode()).hexdigest()[:16]
