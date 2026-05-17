"""Per-site / per-intron query API.

Given an N-variant construct and a specific splice site (or an intron defined
by a donor + acceptor position), return per-context probabilities and the
additive-null residual:

    expected = Σ p(mut_i) − (N−1)·p(ref)
    residual = p(event) − expected

For an intron the same residual is computed against the canonical PSI estimate

    psi(ctx) = p_acceptor(ctx) × p_donor(ctx)

so paper validations (FAS exon 6, RON / MST1R exon 11) collapse to a single
one-liner against the engine.

The auto-derivation helper picks the smallest annotated (donor, acceptor)
pair that brackets the variant positions — strand-agnostic.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SiteQueryResult:
    """Per-context probabilities + additive residual at a single splice site."""
    position: int
    site_type: str                  # 'acceptor' or 'donor'
    annotated: bool
    contexts: dict                  # {'ref': p, 'mut1': p, …, 'event': p} (NaN where missing)
    expected: float                 # Σ p_mut_i − (N−1)·p_ref
    residual: float                 # p_event − expected

    @property
    def event(self) -> float: return float(self.contexts.get("event", np.nan))
    @property
    def ref(self)   -> float: return float(self.contexts.get("ref",   np.nan))


@dataclass(frozen=True)
class IntronQueryResult:
    """Per-context acceptor + donor probabilities + two PSI estimates + residuals.

    Two PSI conventions are computed at every call — pick whichever the
    downstream analysis wants:

    ``psi``        — independence-product PSI: ``p_acceptor(ctx) × p_donor(ctx)``.
                     Bounded [0, 1]; conservative when both sites must fire.
    ``psi_mean``   — mean per-site activity: ``(p_acceptor(ctx) + p_donor(ctx)) / 2``.
                     Matches the legacy figure1 exon_prob formula (modulo the
                     cryptic-residual subtraction that contributes negligibly
                     on benchmark data) and correlates ~2× higher than ``psi``
                     against the FAS Julien 2016 experimental epistasis.

    Each PSI has its own additive-null residual:
        psi_expected      = Σ PSI(mut_i) − (N−1)·PSI(ref)
        psi_residual      = PSI(event) − psi_expected
        psi_mean_expected = Σ PSI_mean(mut_i) − (N−1)·PSI_mean(ref)
        psi_mean_residual = PSI_mean(event) − psi_mean_expected
    """
    donor_pos: int
    acceptor_pos: int
    donor:    SiteQueryResult       # per-context donor probabilities
    acceptor: SiteQueryResult       # per-context acceptor probabilities
    psi:      dict                  # acc × don PSI per context
    psi_expected: float
    psi_residual: float
    psi_mean:      dict             # (acc + don) / 2 per context
    psi_mean_expected: float
    psi_mean_residual: float


# ─────────────────────────────────────────────────────────────────────────────
# Pure functions on a wide site_table
# ─────────────────────────────────────────────────────────────────────────────
def _context_keys(n_variants: int) -> list[str]:
    return ["ref"] + [f"mut{i+1}" for i in range(n_variants)] + ["event"]


def query_site_from_table(
    site_table_wide: pd.DataFrame,
    position: int,
    site_type: str,
    n_variants: int,
) -> SiteQueryResult:
    """Read one site's per-context probabilities from a wide site_table and
    derive the additive-null residual. NaN-fills any missing context with a
    warning so caller can detect and handle gracefully."""
    if site_type not in ("donor", "acceptor"):
        raise ValueError(f"site_type must be 'donor' or 'acceptor', got {site_type!r}")
    keys = _context_keys(n_variants)

    rows = site_table_wide[
        (site_table_wide.position == position)
        & (site_table_wide.site_type == site_type)
    ]
    if rows.empty:
        warnings.warn(
            f"Site {position}/{site_type} not present in site_table — "
            "returning NaN-filled result.",
            stacklevel=2,
        )
        return SiteQueryResult(
            position=int(position), site_type=site_type, annotated=False,
            contexts={k: float("nan") for k in keys},
            expected=float("nan"), residual=float("nan"),
        )

    row = rows.iloc[0]
    contexts: dict = {}
    missing: list[str] = []
    for k in keys:
        v = row[k] if k in rows.columns else float("nan")
        if pd.isna(v):
            missing.append(k)
        contexts[k] = float(v) if pd.notna(v) else float("nan")
    if missing:
        warnings.warn(
            f"Site {position}/{site_type}: missing prediction in contexts "
            f"{missing} — NaN-filled.",
            stacklevel=2,
        )

    ref_p   = contexts["ref"]
    mut_ps  = [contexts[f"mut{i+1}"] for i in range(n_variants)]
    event_p = contexts["event"]
    # additive-null residual; NaN propagates from any missing context
    expected = sum(mut_ps) - (n_variants - 1) * ref_p
    residual = event_p - expected
    return SiteQueryResult(
        position=int(position),
        site_type=site_type,
        annotated=bool(row.get("annotated", False)),
        contexts=contexts,
        expected=float(expected),
        residual=float(residual),
    )


def query_intron_from_table(
    site_table_wide: pd.DataFrame,
    donor_pos: int,
    acceptor_pos: int,
    n_variants: int,
) -> IntronQueryResult:
    """Read both ends of an intron, derive both PSI estimates and their residuals.

    Returns:
      psi[ctx]       = p_acceptor(ctx) × p_donor(ctx)         (independence product)
      psi_mean[ctx]  = (p_acceptor(ctx) + p_donor(ctx)) / 2   (legacy figure1 exon_prob)
    """
    d = query_site_from_table(site_table_wide, donor_pos,    "donor",    n_variants)
    a = query_site_from_table(site_table_wide, acceptor_pos, "acceptor", n_variants)
    keys = _context_keys(n_variants)

    psi      = {k: d.contexts[k] * a.contexts[k]       for k in keys}
    psi_mean = {k: (d.contexts[k] + a.contexts[k]) / 2 for k in keys}

    def _residual(p):
        ref_p   = p["ref"]
        event_p = p["event"]
        exp = sum(p[f"mut{i+1}"] for i in range(n_variants)) - (n_variants - 1) * ref_p
        return float(exp), float(event_p - exp)

    psi_expected,      psi_residual      = _residual(psi)
    psi_mean_expected, psi_mean_residual = _residual(psi_mean)

    return IntronQueryResult(
        donor_pos=int(donor_pos),
        acceptor_pos=int(acceptor_pos),
        donor=d,
        acceptor=a,
        psi=psi,
        psi_expected=psi_expected,
        psi_residual=psi_residual,
        psi_mean=psi_mean,
        psi_mean_expected=psi_mean_expected,
        psi_mean_residual=psi_mean_residual,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Auto-derive the bracketing (donor, acceptor) pair
# ─────────────────────────────────────────────────────────────────────────────
def auto_derive_intron(
    annotated_donors:    Iterable[int],
    annotated_acceptors: Iterable[int],
    variant_positions:   Iterable[int],
) -> Tuple[int, int]:
    """Return (donor_pos, acceptor_pos) — the smallest annotated pair that
    brackets every variant. Strand-agnostic: checks ``min ≤ var ≤ max``,
    not order, so forward- and reverse-strand exons both work.

    Raises ``ValueError`` if no bracketing pair exists.
    """
    vp = [int(p) for p in variant_positions]
    if not vp:
        raise ValueError("auto_derive_intron: no variant positions given.")
    vlo, vhi = min(vp), max(vp)
    donors    = [int(d) for d in annotated_donors]
    acceptors = [int(a) for a in annotated_acceptors]

    best: Optional[Tuple[int, int, int]] = None   # (span, donor, acceptor)
    for d in donors:
        for a in acceptors:
            lo, hi = (d, a) if d < a else (a, d)
            if lo <= vlo and vhi <= hi:
                span = hi - lo
                if best is None or span < best[0]:
                    best = (span, d, a)
    if best is None:
        raise ValueError(
            f"auto_derive_intron: no annotated (donor, acceptor) pair brackets "
            f"variants in [{vlo}, {vhi}] — pass donor_pos and acceptor_pos explicitly."
        )
    return best[1], best[2]
