"""
Splicing-site primitives.

`extract_site_table` flattens a multi-context splicing prediction (the
``adjoin_splicing_outcomes`` output from geney) into a long-format DataFrame
with one row per (position, site_type, context). This is the canonical
data structure for everything downstream — missplicing classification,
epistasis residuals, plotting.

`classify_missplicing` produces the four-event missplicing description used
in the original Oncosplice paper:

- **missed_donor**     — annotated donor whose probability dropped by ≥ Δ
- **missed_acceptor**  — annotated acceptor whose probability dropped by ≥ Δ
- **discovered_donor** — unannotated position where donor probability rose by ≥ Δ
- **discovered_acceptor** — unannotated position where acceptor probability rose by ≥ Δ
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

DEFAULT_DELTA_THRESHOLD = 0.25  # Oncosplice paper default


def extract_site_table(
    splicing_results: pd.DataFrame,
    contexts: List[str],
) -> pd.DataFrame:
    """Convert geney's wide multi-index splicing DataFrame to a long table.

    Parameters
    ----------
    splicing_results
        Output of ``geney.splicing.adjoin_splicing_outcomes`` — a DataFrame
        whose index is genomic position and whose columns are
        ``(site_type, metric)`` MultiIndex with ``site_type ∈ {donors, acceptors}``
        and ``metric ∈ {ref_prob, <ctx>_prob, annotated, ...}``.
    contexts
        Labels of the contexts to include. Always pass ``ref`` first plus any
        of ``event``, ``mut1``, ``mut2``.

    Returns
    -------
    pd.DataFrame
        Long-format table with columns ``[position, site_type, context, prob,
        annotated]`` — one row per (position, site_type, context).  Reference
        is reported as context ``"ref"``.
    """
    rows = []
    for site_type in ("donors", "acceptors"):
        if site_type not in splicing_results.columns.get_level_values(0):
            continue
        sub = splicing_results[site_type].copy()
        # ``annotated`` is broadcast, ``ref_prob`` is the wild-type column
        annotated = sub.get("annotated", pd.Series(False, index=sub.index)).fillna(False).astype(bool)
        for ctx in contexts:
            col = "ref_prob" if ctx == "ref" else f"{ctx}_prob"
            if col not in sub.columns:
                continue
            probs = sub[col].astype(float).fillna(0.0)
            tmp = pd.DataFrame({
                "position": sub.index.values,
                "site_type": site_type[:-1],   # 'donor' / 'acceptor'
                "context": ctx,
                "prob": probs.values,
                "annotated": annotated.values,
            })
            rows.append(tmp)
    if not rows:
        return pd.DataFrame(columns=["position", "site_type", "context", "prob", "annotated"])
    out = pd.concat(rows, ignore_index=True)
    return out


def site_table_wide(site_table: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long site_table to one row per (position, site_type) with
    columns for each context's probability.
    """
    if site_table.empty:
        return site_table
    wide = site_table.pivot_table(
        index=["position", "site_type", "annotated"],
        columns="context",
        values="prob",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    return wide


@dataclass(frozen=True)
class MissplicingEvent:
    """A single per-site change in splicing probability."""
    position: int
    site_type: str        # 'donor' or 'acceptor'
    event_type: str       # 'missed_donor' / 'missed_acceptor' / 'discovered_donor' / 'discovered_acceptor'
    ref_prob: float
    var_prob: float
    delta: float

    def as_dict(self) -> dict:
        return {
            "position": int(self.position),
            "site_type": self.site_type,
            "event_type": self.event_type,
            "ref_prob": float(self.ref_prob),
            "var_prob": float(self.var_prob),
            "delta": float(self.delta),
        }


def classify_missplicing(
    site_table: pd.DataFrame,
    *,
    context: str = "event",
    threshold: float = DEFAULT_DELTA_THRESHOLD,
) -> List[MissplicingEvent]:
    """Identify the per-site missplicing events for a given variant context.

    Rules (verbatim from Oncosplice paper):
    - At an *annotated* site, a drop of ``ref - var ≥ threshold`` ⇒ missed_<type>.
    - At an *unannotated* site, a rise of ``var - ref ≥ threshold`` ⇒ discovered_<type>.

    Returns events sorted by descending |delta|.
    """
    wide = site_table_wide(site_table)
    if wide.empty or context not in wide.columns:
        return []
    if "ref" not in wide.columns:
        raise ValueError("site_table must include the 'ref' context.")

    events: List[MissplicingEvent] = []
    for _, row in wide.iterrows():
        ref_p = float(row["ref"]) if pd.notna(row["ref"]) else 0.0
        var_p = float(row[context]) if pd.notna(row[context]) else 0.0
        delta = var_p - ref_p
        annotated = bool(row["annotated"])
        site_type = str(row["site_type"])

        if annotated and delta <= -threshold:
            events.append(MissplicingEvent(
                position=int(row["position"]),
                site_type=site_type,
                event_type=f"missed_{site_type}",
                ref_prob=ref_p, var_prob=var_p, delta=delta,
            ))
        elif (not annotated) and delta >= threshold:
            events.append(MissplicingEvent(
                position=int(row["position"]),
                site_type=site_type,
                event_type=f"discovered_{site_type}",
                ref_prob=ref_p, var_prob=var_p, delta=delta,
            ))

    events.sort(key=lambda e: abs(e.delta), reverse=True)
    return events


def missplicing_to_dict(events: List[MissplicingEvent]) -> Dict[str, Dict[int, dict]]:
    """Group events into the {missed_donors, missed_acceptors, discovered_donors,
    discovered_acceptors} dict shape that downstream code (and the original paper
    figures) consume.
    """
    out: Dict[str, Dict[int, dict]] = {
        "missed_donors": {},
        "missed_acceptors": {},
        "discovered_donors": {},
        "discovered_acceptors": {},
    }
    bucket = {
        "missed_donor": "missed_donors",
        "missed_acceptor": "missed_acceptors",
        "discovered_donor": "discovered_donors",
        "discovered_acceptor": "discovered_acceptors",
    }
    for e in events:
        out[bucket[e.event_type]][e.position] = {
            "ref_prob": e.ref_prob,
            "var_prob": e.var_prob,
            "delta": e.delta,
        }
    return out


def max_splicing_delta(site_table: pd.DataFrame, context: str = "event") -> float:
    """Largest signed change in splicing probability between ref and ``context``.

    ``max(|var - ref|)`` retaining sign — useful as a single-number summary
    of how disruptive a variant is at the splicing level (the
    ``missplicing`` scalar in the result objects).
    """
    wide = site_table_wide(site_table)
    if wide.empty or "ref" not in wide.columns or context not in wide.columns:
        return 0.0
    diffs = (wide[context].astype(float).fillna(0) - wide["ref"].astype(float).fillna(0))
    if diffs.empty:
        return 0.0
    return float(diffs.iloc[diffs.abs().argmax()])
