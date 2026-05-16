"""
Visualization — make residuals at splice sites legible at a glance.

Three plot families:

1. :func:`plot_single_missplicing` — single-variant landscape; shows ref and
   variant donor/acceptor probabilities along the genome with detected events
   annotated.
2. :func:`plot_pair_residuals` — *the* double-variant figure: per splice site,
   show the four context probabilities (ref/mut1/mut2/event), the additive
   expectation, and the observed event with a colored arrow indicating the
   residual.  Synergistic sites are tinted red, antagonistic blue,
   non-epistatic gray.
3. :func:`plot_pair_landscape` — full-window genomic view: ref, mut1, mut2,
   event probability tracks stacked; shaded regions over splice sites whose
   residual exceeds threshold.
4. :func:`plot_pair_summary` — composite (top: landscape; middle: per-site
   waterfall; bottom: residual bars).

All functions return ``(fig, axes)`` so callers can further customize.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from .results import DoubleVariantResult, SingleVariantResult


# ---- Color scheme (legible, color-blind friendly, consistent across plots) --
_COLORS = {
    "ref":           "#444444",   # WT / reference
    "mut1":          "#1f77b4",   # constituent variant 1 (solid)
    "mut2":          "#1f77b4",   # constituent variant 2 (same colour, distinguished by hatch)
    "event":         "#2ca02c",   # observed joint
    "expected":      "#777777",   # additive null prediction (hatched)
    # 4 mechanism classes
    "deletion_synergy": "#d62728",   # red    — both preserve, joint destroys
    "cryptic_synergy":  "#ff7f0e",   # orange — both silent, joint creates novel site
    "rescue":           "#0072b2",   # blue   — one disrupts, joint restores
    "cryptic_rescue":   "#56b4e9",   # light blue — one creates cryptic, joint silences
    "non-epistatic":    "#bbbbbb",
    # legacy aliases (so any stray plotting code that still says "synergistic"
    # gets a sensible colour rather than KeyError)
    "synergistic":   "#d62728",
    "compounding":   "#cc79a7",
    "antagonistic":  "#9467bd",
    "donor":         "#1f77b4",
    "acceptor":      "#d62728",
}


def _import_mpl():
    import matplotlib.pyplot as plt
    return plt


# ---------------------------------------------------------------------------
# Single-variant
# ---------------------------------------------------------------------------
def plot_single_missplicing(
    result: "SingleVariantResult",
    *,
    threshold: float = 0.25,
    figsize: tuple = (10, 4),
):
    """Bar plot of |Δprob| at each splice site, colored by event type."""
    plt = _import_mpl()
    df = result.missplicing.to_dataframe()
    if df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No missplicing detected at threshold "
                          f"{threshold}", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig, ax

    fig, ax = plt.subplots(figsize=figsize)
    palette = {
        "missed_donor":        _COLORS["donor"],
        "missed_acceptor":     _COLORS["acceptor"],
        "discovered_donor":    _COLORS["donor"],
        "discovered_acceptor": _COLORS["acceptor"],
    }
    hatch_map = {
        "missed_donor":        "",
        "missed_acceptor":     "",
        "discovered_donor":    "//",
        "discovered_acceptor": "//",
    }
    for et, sub in df.groupby("event_type"):
        ax.bar(sub.position, sub.delta,
               color=palette.get(et, "#888"),
               hatch=hatch_map.get(et, ""),
               edgecolor="black", linewidth=0.5,
               label=et.replace("_", " "))
    ax.axhline(0, color="black", lw=0.5)
    ax.axhline(threshold, color="gray", ls="--", lw=0.5)
    ax.axhline(-threshold, color="gray", ls="--", lw=0.5)
    ax.set_xlabel("Genomic position")
    ax.set_ylabel("Δprob (variant − reference)")
    ax.set_title(f"{result.mut_id}  —  {result.splicing_engine}")
    ax.legend(loc="best", fontsize=8, frameon=False)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Double-variant — per-site residual figure
# ---------------------------------------------------------------------------
def plot_pair_residuals(
    result: "DoubleVariantResult",
    *,
    only_active: bool = True,
    activity_min: float = 0.1,
    figsize: Optional[tuple] = None,
    annotate_residual: bool = True,
):
    """Per-splice-site view of (ref, mut1, mut2, event) with the additive
    expectation and the residual highlighted.

    For each site that passes ``activity_min`` (max over the four contexts) we
    draw five points along a small vertical track:

        ref   ─●        mut1 ─●         mut2 ─●        expected ─◇         event ─●

    A colored arrow connects ``expected`` to ``event``; arrow color encodes
    the per-site classification (red=synergistic, blue=antagonistic, gray=non-
    epistatic).  When ``only_active=True`` (default) we drop sites where all
    four probabilities are below ``activity_min``.
    """
    plt = _import_mpl()
    sites = result.site_residuals.copy()
    if only_active:
        max_prob = sites[["ref", "mut1", "mut2", "event"]].abs().max(axis=1)
        sites = sites[max_prob >= activity_min]
    sites = sites.sort_values(["site_type", "position"]).reset_index(drop=True)
    n = len(sites)

    if n == 0:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, "No active splice sites in window.",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig, ax

    if figsize is None:
        figsize = (max(8, 0.6 * n + 4), 5)
    fig, ax = plt.subplots(figsize=figsize)

    x_offsets = {"ref": -0.30, "mut1": -0.10, "mut2": 0.10, "expected": 0.25, "event": 0.40}

    for i, row in sites.iterrows():
        x_center = i

        # 1. Reference, mut1, mut2 dots
        for ctx in ("ref", "mut1", "mut2"):
            ax.plot(x_center + x_offsets[ctx], row[ctx],
                    marker="o", markersize=6, color=_COLORS[ctx], linestyle="")

        # 2. Expected (open diamond) and event (filled square)
        ax.plot(x_center + x_offsets["expected"], row["expected"],
                marker="D", markersize=8, markerfacecolor="white",
                markeredgecolor=_COLORS["expected"], linewidth=1.2)
        ax.plot(x_center + x_offsets["event"], row["event"],
                marker="s", markersize=8, color=_COLORS["event"])

        # 3. Arrow expected → event, colored by classification
        cls = row["classification"]
        arrow_color = _COLORS.get(cls, _COLORS["non-epistatic"])
        if abs(row["residual"]) >= 0.005:
            ax.annotate(
                "",
                xy=(x_center + x_offsets["event"], row["event"]),
                xytext=(x_center + x_offsets["expected"], row["expected"]),
                arrowprops=dict(arrowstyle="->", color=arrow_color, lw=1.8),
            )
        # 4. Site classification background tint
        if cls != "non-epistatic":
            ax.axvspan(x_center - 0.45, x_center + 0.55,
                       color=arrow_color, alpha=0.06, zorder=-2)

        # 5. Residual annotation
        if annotate_residual and abs(row["residual"]) >= 0.05:
            ax.text(x_center + x_offsets["event"] + 0.05,
                    row["event"],
                    f"r={row['residual']:+.2f}",
                    fontsize=7, va="center", color=arrow_color)

    # X-axis: site labels
    labels = [
        f"{int(r.position)}\n{'●' if r.annotated else '○'}{r.site_type[0].upper()}"
        for r in sites.itertuples()
    ]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlim(-0.7, n - 0.3)
    ax.set_ylim(-0.05, max(1.05, sites[["ref","mut1","mut2","event","expected"]].max().max() + 0.05))
    ax.set_ylabel("Splice probability")
    ax.set_xlabel("Splice site (position; ●=annotated, ○=cryptic; D=donor, A=acceptor)")
    ax.set_title(
        f"{result.epistasis_id}  —  pair: {result.pair_classification}  "
        f"(score residual = {result.score_residual:+.2f})"
    )

    # Legend (single set of proxy artists)
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_COLORS["ref"],   label="ref",       markersize=6),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_COLORS["mut1"],  label="mut1",      markersize=6),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_COLORS["mut2"],  label="mut2",      markersize=6),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="white",
               markeredgecolor=_COLORS["expected"], label="expected (additive)", markersize=8),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=_COLORS["event"], label="event (joint)", markersize=8),
        Line2D([0], [0], color=_COLORS["deletion_synergy"], lw=2, label="deletion synergy"),
        Line2D([0], [0], color=_COLORS["cryptic_synergy"],  lw=2, label="cryptic synergy"),
        Line2D([0], [0], color=_COLORS["rescue"],           lw=2, label="rescue"),
        Line2D([0], [0], color=_COLORS["cryptic_rescue"],   lw=2, label="cryptic rescue"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, frameon=False, ncol=2)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Double-variant — full genomic landscape
# ---------------------------------------------------------------------------
def plot_pair_landscape(
    result: "DoubleVariantResult",
    *,
    site_type: str = "donor",
    figsize: tuple = (12, 4),
):
    """Stacked per-context probability tracks across the genomic window.

    One panel for donors, one for acceptors. Probabilities for the four
    contexts are plotted as filled curves; sites whose |residual| ≥ threshold
    are marked with a colored vertical band.
    """
    plt = _import_mpl()
    fig, ax = plt.subplots(figsize=figsize)

    df = result.site_table[result.site_table.site_type == site_type]
    if df.empty:
        ax.text(0.5, 0.5, f"No {site_type} sites in window.",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig, ax

    pivot = df.pivot_table(index="position", columns="context",
                           values="prob", aggfunc="first").fillna(0)
    pivot = pivot.sort_index()

    for ctx in ("ref", "mut1", "mut2", "event"):
        if ctx in pivot.columns:
            ax.plot(pivot.index, pivot[ctx], color=_COLORS[ctx],
                    label=ctx, lw=1.0, alpha=0.9)

    # Mark mutation positions with vertical lines
    ax.axvline(int(result.mut1_id.split(":")[2]), color=_COLORS["mut1"],
               ls=":", lw=1, alpha=0.6)
    ax.axvline(int(result.mut2_id.split(":")[2]), color=_COLORS["mut2"],
               ls=":", lw=1, alpha=0.6)

    # Highlight epistatic sites
    epi = result.site_residuals[
        (result.site_residuals.site_type == site_type)
        & (result.site_residuals.classification != "non-epistatic")
    ]
    for _, row in epi.iterrows():
        ax.axvspan(row.position - 1, row.position + 1,
                   color=_COLORS[row.classification], alpha=0.18, zorder=-2)

    ax.set_xlabel("Genomic position")
    ax.set_ylabel(f"{site_type} probability")
    ax.set_title(f"{result.epistasis_id} — {site_type} landscape")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.set_ylim(0, max(1.05, pivot.max().max() + 0.05))
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Composite three-panel figure
# ---------------------------------------------------------------------------
def plot_pair_summary(
    result: "DoubleVariantResult",
    *,
    figsize: tuple = (12, 9),
    activity_min: float = 0.1,
):
    """Three-panel composite:

    - top:    donor + acceptor landscape across the genomic window
    - middle: the per-site residual figure (the headline plot)
    - bottom: bar chart of Oncosplice scores in each of the four contexts
    """
    plt = _import_mpl()
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.4, 0.7], hspace=0.45,
                          width_ratios=[1, 1])

    # Top row — landscape (donor and acceptor)
    ax_d = fig.add_subplot(gs[0, 0])
    ax_a = fig.add_subplot(gs[0, 1], sharey=ax_d)
    for ax, st in [(ax_d, "donor"), (ax_a, "acceptor")]:
        df = result.site_table[result.site_table.site_type == st]
        if df.empty:
            ax.text(0.5, 0.5, f"no {st}s", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_axis_off()
            continue
        pivot = df.pivot_table(index="position", columns="context",
                               values="prob", aggfunc="first").fillna(0).sort_index()
        for ctx in ("ref", "mut1", "mut2", "event"):
            if ctx in pivot.columns:
                ax.plot(pivot.index, pivot[ctx], color=_COLORS[ctx],
                        label=ctx, lw=1)
        epi = result.site_residuals[
            (result.site_residuals.site_type == st)
            & (result.site_residuals.classification != "non-epistatic")
        ]
        for _, row in epi.iterrows():
            ax.axvspan(row.position - 1, row.position + 1,
                       color=_COLORS[row.classification], alpha=0.18, zorder=-2)
        ax.set_title(f"{st} probabilities")
        ax.set_xlabel("position")
        ax.set_ylabel("prob")
        if st == "donor":
            ax.legend(loc="upper right", frameon=False, fontsize=7, ncol=2)

    # Middle row — per-site residuals (full width)
    ax_mid = fig.add_subplot(gs[1, :])
    sites = result.site_residuals.copy()
    max_prob = sites[["ref", "mut1", "mut2", "event"]].abs().max(axis=1)
    sites = sites[max_prob >= activity_min].sort_values(
        ["site_type", "position"]
    ).reset_index(drop=True)

    if not sites.empty:
        n = len(sites)
        x_offsets = {"ref": -0.30, "mut1": -0.10, "mut2": 0.10,
                     "expected": 0.25, "event": 0.40}
        for i, row in sites.iterrows():
            for ctx in ("ref", "mut1", "mut2"):
                ax_mid.plot(i + x_offsets[ctx], row[ctx], "o",
                            color=_COLORS[ctx], markersize=5)
            ax_mid.plot(i + x_offsets["expected"], row["expected"], "D",
                        markerfacecolor="white",
                        markeredgecolor=_COLORS["expected"], markersize=7)
            ax_mid.plot(i + x_offsets["event"], row["event"], "s",
                        color=_COLORS["event"], markersize=7)
            cls = row["classification"]
            arrow_color = _COLORS.get(cls, _COLORS["non-epistatic"])
            if abs(row["residual"]) >= 0.005:
                ax_mid.annotate(
                    "", xy=(i + x_offsets["event"], row["event"]),
                    xytext=(i + x_offsets["expected"], row["expected"]),
                    arrowprops=dict(arrowstyle="->", color=arrow_color, lw=1.5),
                )
            if cls != "non-epistatic":
                ax_mid.axvspan(i - 0.45, i + 0.55, color=arrow_color, alpha=0.06, zorder=-2)
        ax_mid.set_xticks(range(n))
        ax_mid.set_xticklabels(
            [f"{int(r.position)}\n{r.site_type[0].upper()}" for r in sites.itertuples()],
            fontsize=8,
        )
        ax_mid.set_xlim(-0.7, n - 0.3)
        ax_mid.set_ylabel("splice prob")
        ax_mid.set_title("Per-site epistasis: ref · mut1 · mut2 · expected ◇ ↦ ■ event")
    else:
        ax_mid.text(0.5, 0.5, "no active splice sites", ha="center", va="center",
                    transform=ax_mid.transAxes)
        ax_mid.set_axis_off()

    # Bottom row — Oncosplice scores per context + score residual
    ax_score = fig.add_subplot(gs[2, 0])
    ctxs = ["ref", "mut1", "mut2", "event"]
    vals = [result.oncosplice_scores.get(c, 0.0) for c in ctxs]
    ax_score.bar(ctxs, vals, color=[_COLORS[c] for c in ctxs])
    ax_score.set_ylabel("Oncosplice score")
    ax_score.set_title("Functional divergence per context")

    ax_resid = fig.add_subplot(gs[2, 1])
    score_expected = (result.oncosplice_scores.get("mut1", 0)
                      + result.oncosplice_scores.get("mut2", 0)
                      - result.oncosplice_scores.get("ref", 0))
    bars = ["expected (additive)", "observed (joint)", "residual"]
    score_event = result.oncosplice_scores.get("event", 0)
    bar_vals = [score_expected, score_event, result.score_residual]
    bar_colors = [_COLORS["expected"], _COLORS["event"],
                  _COLORS.get(result.pair_classification, _COLORS["non-epistatic"])]
    ax_resid.bar(bars, bar_vals, color=bar_colors)
    ax_resid.axhline(0, color="black", lw=0.5)
    ax_resid.set_ylabel("score")
    ax_resid.set_title(
        f"Score-level residual: {result.score_residual:+.2f}\n"
        f"pair: {result.pair_classification}"
    )

    fig.suptitle(
        f"{result.epistasis_id}   ·   distance {result.distance} nt   ·   "
        f"engine: {result.splicing_engine}",
        fontsize=11,
    )
    return fig, (ax_d, ax_a, ax_mid, ax_score, ax_resid)


# ---------------------------------------------------------------------------
# Case-study figure — single-pair mechanistic summary
# ---------------------------------------------------------------------------

_MECHANISTIC_INTERPRETATIONS = {
    "deletion_synergy":
        "Annotated splice site is preserved by each single variant on its own, "
        "but destroyed only when both mutations co-occur. The joint effect is "
        "an emergent loss that neither variant predicts alone.",
    "cryptic_synergy":
        "No annotated splice site at this position, and neither single variant "
        "creates one. Together the two variants generate a novel cryptic site — "
        "a discrete gain-of-function that requires both mutations.",
    "rescue":
        "Annotated splice site is severely disrupted by one of the two single "
        "variants. The joint variant restores splicing close to wild-type — "
        "one mutation cancels the splicing damage of the other.",
    "cryptic_rescue":
        "One single variant creates a strong cryptic splice site where none "
        "exists in WT. The joint variant silences that cryptic — the second "
        "mutation suppresses the splicing aberration caused by the first.",
    "non-epistatic":
        "No interaction detected at this site: neither single has a meaningful "
        "effect, the joint matches the additive expectation, or the singles "
        "are dominant enough that the joint adds nothing new.",
    # legacy labels (kept so old notebooks still render a sensible caption)
    "synergistic":
        "Joint variant produces a splicing change LARGER than the sum of "
        "individual effects (legacy label — see deletion_synergy / "
        "cryptic_synergy in the current taxonomy).",
    "compounding":
        "Both variants push splicing in the same direction; together they "
        "magnify each other (legacy label).",
    "antagonistic":
        "Joint variant has a SMALLER splicing effect than the additive sum of "
        "singles (legacy label).",
}


def _format_mutation(mut_id: str) -> str:
    """Compact display name: 'KRAS:12:25227344:A:T' → 'KRAS 25227344 A>T'."""
    try:
        gene, _chrm, pos, ref, alt = mut_id.split(":")
        return f"{gene} {pos} {ref}>{alt}"
    except Exception:
        return mut_id


def plot_pair_case_study(
    result: "DoubleVariantResult",
    *,
    style: str = "bars",
    **kwargs,
):
    """Nature-style mechanistic figure for one double-variant pair.

    Two display styles:

    - ``style="bars"`` (default) — one grouped-bar cluster per affected
      splice site, with WT / mut1 / mut2 grouped tight, then a small gap,
      then expected / joint. Best when the reader wants to see absolute
      probability at each context.
    - ``style="slope"`` — one line per splice site, X-axis is the 4
      contexts (WT → mut1 → mut2 → joint). Best when the reader wants to
      see the *trajectory* — affected sites move, normal sites are flat.

    Both styles include flanking non-affected sites for local context.

    Parameters
    ----------
    style : {"bars", "slope"}
        Visualization style; default ``"bars"``.
    **kwargs
        Forwarded to the style-specific implementation. See
        :func:`_plot_case_study_bars` and :func:`_plot_case_study_slope`
        for the supported keyword arguments.
    """
    if style == "bars":
        return _plot_case_study_bars(result, **kwargs)
    if style == "slope":
        return _plot_case_study_slope(result, **kwargs)
    raise ValueError(f"style must be 'bars' or 'slope'; got {style!r}")


# ── Shared site-selection helper for both styles ─────────────────────────

_NATURE_RC = {
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":          7,
    "axes.linewidth":     0.6,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "xtick.major.size":   2.0,
    "ytick.major.size":   2.0,
    "xtick.major.pad":    2.0,
    "ytick.major.pad":    2.0,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "legend.frameon":     False,
}


def _select_case_study_sites(result, *, activity_min, annotated_only,
                               flanking_sites, max_sites):
    """Filter + order the sites for a case-study figure.

    Returns a DataFrame sorted in mRNA direction. Always anchors the
    panel with the **nearest annotated donor and nearest annotated
    acceptor** (so the reader can see the canonical splicing landscape
    of the local exon), plus all affected sites and their nearest active
    neighbors, capped at ``max_sites``.
    """
    sites = result.site_residuals.copy()
    if annotated_only:
        sites = sites[sites.annotated]
    if not sites.empty:
        max_prob = sites[["ref", "mut1", "mut2", "event"]].abs().max(axis=1)
        sites = sites[max_prob >= activity_min]
    if sites.empty:
        return sites

    # mRNA-direction sort
    pos_series = result.site_table.position.drop_duplicates()
    is_negative_strand = (
        len(pos_series) >= 2 and pos_series.iloc[0] > pos_series.iloc[-1]
    )
    sites = sites.sort_values("position",
                              ascending=not is_negative_strand).reset_index(drop=True)

    # ── 1. Always-keep set: affected sites + ALL strong annotated anchors
    keep = set(sites.index[sites.classification != "non-epistatic"].tolist())

    affected_positions = sites.loc[list(keep), "position"].tolist() if keep else []
    if not affected_positions:
        affected_positions = [result.central_position]
    aff_lo, aff_hi = min(affected_positions), max(affected_positions)

    # Show EVERY annotated splice site that (a) carries real signal
    # (max prob ≥ 0.5 in any context) and (b) sits within or just outside
    # the affected-position window. This guarantees the reader sees the
    # canonical donor + acceptor landscape that defines the local exons.
    annotated_active = sites[sites.annotated &
                              (sites[["ref","mut1","mut2","event"]].max(axis=1) >= 0.5)].copy()
    keep.update(annotated_active.index.tolist())

    # If, after taking strong-annotated sites, we have no annotated site of
    # one type (donor or acceptor) on either side of the affected region,
    # fall back to the nearest annotated site of that type even if weak.
    for st in ("donor", "acceptor"):
        already = sites.loc[list(keep)]
        if (already.site_type == st).any() and already[already.site_type == st].annotated.any():
            continue
        cand = sites[(sites.site_type == st) & sites.annotated]
        if cand.empty:
            continue
        cand_dist = (cand.position - (aff_lo + aff_hi) / 2).abs()
        keep.add(cand_dist.idxmin())

    # ── 2. Flanking active neighbors on each side of affected sites
    affected_idx = sites.index[sites.classification != "non-epistatic"].tolist()
    for ai in affected_idx:
        for j in range(max(0, ai - flanking_sites),
                       min(len(sites), ai + flanking_sites + 1)):
            keep.add(j)

    sites = sites.iloc[sorted(keep)].reset_index(drop=True)

    if max_sites is not None and len(sites) > max_sites:
        # Always preserve affected + the canonical anchors; drop weakest extras
        affected_mask = sites.classification != "non-epistatic"
        anchor_mask = sites.annotated & (sites[["ref","mut1","mut2","event"]].max(axis=1) >= 0.5)
        priority = (affected_mask.astype(int) * 10 + anchor_mask.astype(int) * 5
                    + sites[["ref","mut1","mut2","event"]].max(axis=1))
        keep_idx = priority.sort_values(ascending=False).head(max_sites).index
        sites = sites.loc[keep_idx].sort_values(
            "position", ascending=not is_negative_strand,
        ).reset_index(drop=True)
    return sites


# ── Style: grouped bars ──────────────────────────────────────────────────

def _plot_case_study_bars(
    result: "DoubleVariantResult",
    *,
    activity_min: float = 0.10,
    figsize: Optional[Tuple[float, float]] = None,
    annotated_only: bool = False,
    flanking_sites: int = 1,
    max_sites: Optional[int] = 6,
):
    """Grouped-bar mechanistic figure (default 'bars' style).

    Per splice site: 5 bars per cluster — WT, mut1, mut2 (paired, same
    blue; mut2 with thin dark outline), a clear gap, then expected
    (hatched outline) and joint (green). Tells the mechanism in absolute
    probability values per context.
    """
    import matplotlib as _mpl
    plt = _import_mpl()
    from matplotlib.patches import Patch

    sites = _select_case_study_sites(
        result, activity_min=activity_min, annotated_only=annotated_only,
        flanking_sites=flanking_sites, max_sites=max_sites,
    )
    if sites.empty:
        fig, ax = plt.subplots(figsize=figsize or (3.5, 2.3))
        ax.text(0.5, 0.5, "No active splice sites in window.",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=8, color="#666")
        ax.set_axis_off()
        return fig, ax

    n_sites = len(sites)
    contexts = ["ref", "mut1", "mut2", "expected", "event"]
    [_COLORS[c] for c in contexts]

    if figsize is None:
        width = max(3.5, 0.95 * max(1, n_sites) + 1.0)
        figsize = (min(width, 7.2), 2.6)

    with _mpl.rc_context(_NATURE_RC):
        fig, ax = plt.subplots(figsize=figsize)

        # ── Grouped bars: {WT, mut1, mut2}  | gap |  {expected, joint} ─
        bar_w = 0.18
        offsets = np.array([-2.4, -1.4, -0.4, 1.1, 2.1]) * bar_w
        cluster_span = (offsets.max() - offsets.min()) + bar_w
        cluster_gap = cluster_span + 1.20
        site_xs = np.arange(n_sites) * cluster_gap

        # Color/style per context (Nature-clean)
        ctx_style = {
            "ref":      dict(facecolor=_COLORS["ref"],   edgecolor="none"),
            "mut1":     dict(facecolor=_COLORS["mut1"],  edgecolor="none"),
            "mut2":     dict(facecolor=_COLORS["mut1"],  edgecolor="none"),   # same blue as mut1
            "expected": dict(facecolor="white",          edgecolor="#666666", linewidth=0.7),
            "event":    dict(facecolor=_COLORS["event"], edgecolor="none"),
        }

        for i, row in sites.iterrows():
            xc = site_xs[i]
            for k, ctx in enumerate(contexts):
                val = max(0.0, min(1.0, float(row[ctx])))
                ax.bar(xc + offsets[k], val,
                       width=bar_w * 0.92,
                       zorder=2,
                       **ctx_style[ctx])

        # ── Y axis (subtle gridlines for value reading) ──────────────
        ax.set_ylim(0, 1.08)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0", "", "0.5", "", "1"])
        ax.set_ylabel("Splice probability", fontsize=8)
        for y in (0.25, 0.5, 0.75):
            ax.axhline(y, color="#eeeeee", lw=0.5, zorder=0)

        # ── X axis (clearer site labels, bigger fonts) ───────────────
        ax.set_xlim(site_xs[0] - cluster_gap / 2, site_xs[-1] + cluster_gap / 2)
        ax.set_xticks(site_xs)
        ax.set_xticklabels([], fontsize=0)             # we'll annotate manually

        for i, row in sites.iterrows():
            site_letter = row.site_type[0].upper()      # D / A
            is_affected = row.classification != "non-epistatic"
            is_annotated = bool(row.annotated)
            color = "#000000" if is_affected else "#666666"
            weight = "bold" if is_affected else "normal"
            tag = "" if is_annotated else "*"           # asterisk marks cryptic
            ax.text(
                site_xs[i], -0.04,
                f"{site_letter}{tag} {int(row.position):,}",
                transform=ax.get_xaxis_transform(),
                ha="center", va="top",
                fontsize=7.5, fontweight=weight, color=color,
            )
            # second line — annotated/cryptic in tiny gray
            ax.text(
                site_xs[i], -0.10,
                "annotated" if is_annotated else "cryptic",
                transform=ax.get_xaxis_transform(),
                ha="center", va="top",
                fontsize=6, color="#888888",
                fontstyle="italic",
            )
            if is_affected:
                ax.text(
                    site_xs[i], -0.18,
                    f"{row.classification}",
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="top",
                    fontsize=7, fontweight="bold", color="#000000",
                )
                ax.text(
                    site_xs[i], -0.24,
                    f"r = {row.residual:+.2f}",
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="top",
                    fontsize=6.5, color="#000000",
                )

        # ── Legend (above panel) ─────────────────────────────────────
        legend_handles = [
            Patch(facecolor=_COLORS["ref"],   edgecolor="none",  label="WT"),
            Patch(facecolor=_COLORS["mut1"],  edgecolor="none",  label="mut1"),
            Patch(facecolor=_COLORS["mut1"],  edgecolor="none",  label="mut2"),
            Patch(facecolor="white",          edgecolor="#666666", linewidth=0.7,
                  label="expected"),
            Patch(facecolor=_COLORS["event"], edgecolor="none",  label="joint"),
        ]
        ax.legend(
            handles=legend_handles,
            loc="lower center", bbox_to_anchor=(0.5, 1.0),
            ncol=5, fontsize=7,
            handlelength=1.1, handleheight=1.0, columnspacing=1.2,
            borderaxespad=0.3,
        )

        ax.tick_params(axis="x", length=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()

    return fig, ax


# ── Style: slope / trajectory ────────────────────────────────────────────

def _plot_case_study_slope(
    result: "DoubleVariantResult",
    *,
    activity_min: float = 0.10,
    figsize: Optional[Tuple[float, float]] = None,
    annotated_only: bool = False,
    flanking_sites: int = 1,
    max_sites: Optional[int] = 6,
):
    """Slope-plot mechanistic figure (the 'slope' style).

    One line per splice site, X = (WT, mut1, mut2, joint), Y = probability.
    Expected (additive null) shown as an open marker on the joint X with
    a dashed connector. Non-affected sites rendered as a faint grey ribbon
    so the reader sees the local splicing landscape.
    """
    import matplotlib as _mpl
    plt = _import_mpl()

    sites = _select_case_study_sites(
        result, activity_min=activity_min, annotated_only=annotated_only,
        flanking_sites=flanking_sites, max_sites=max_sites,
    )
    if sites.empty:
        fig, ax = plt.subplots(figsize=figsize or (3.5, 2.3))
        ax.text(0.5, 0.5, "No active splice sites in window.",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=8, color="#666")
        ax.set_axis_off()
        return fig, ax

    contexts = ["ref", "mut1", "mut2", "event"]
    context_labels = ["WT", "mut1", "mut2", "joint"]

    if figsize is None:
        figsize = (3.8, 2.6)

    affected = sites[sites.classification != "non-epistatic"].reset_index(drop=True)
    quiet    = sites[sites.classification == "non-epistatic"].reset_index(drop=True)
    palette = ["#000000", "#d62728", "#0072b2", "#cc79a7", "#2ca02c"]

    with _mpl.rc_context(_NATURE_RC):
        fig, ax = plt.subplots(figsize=figsize)
        xs = np.array([0, 1, 2, 3])

        for _, row in quiet.iterrows():
            ys = [float(row[c]) for c in contexts]
            ax.plot(xs, ys, color="#c8c8c8", lw=0.7, marker="o",
                    markersize=2.2, markeredgecolor="none", zorder=1)

        for i, row in affected.iterrows():
            color = palette[i % len(palette)]
            ys = [float(row[c]) for c in contexts]
            ax.plot(xs, ys, color=color, lw=1.4, marker="o",
                    markersize=4.5, markeredgecolor="white", markeredgewidth=0.6,
                    zorder=3)
            exp = max(0.0, min(1.0, float(row["expected"])))
            ax.plot(3, exp, marker="o", markersize=4.5, markerfacecolor="white",
                    markeredgecolor=color, markeredgewidth=1.0, zorder=4)
            ax.plot([3, 3], [ys[3], exp], color=color, lw=0.8,
                    linestyle="--", zorder=2)
            label = f"{row.site_type[0].upper()} {int(row.position):,}"
            ax.text(3.15, ys[3], label, fontsize=6.5, va="center",
                    color=color, fontweight="bold")
            ax.text(3.15, ys[3] - 0.06,
                    f"{row.classification}, r={row.residual:+.2f}",
                    fontsize=5.5, va="center", color="#000000",
                    fontstyle="italic")

        ax.set_xlim(-0.25, 4.10)
        ax.set_xticks(xs)
        ax.set_xticklabels(context_labels)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_yticklabels(["0", "0.5", "1"])
        ax.set_ylabel("Splice probability", fontsize=7)
        for x in xs:
            ax.axvline(x, color="#eeeeee", lw=0.5, zorder=0)

        if not quiet.empty:
            ax.text(
                0.02, 0.98,
                f"n = {len(quiet)} other sites (grey)",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=5.5, color="#888888",
            )

        from matplotlib.lines import Line2D
        legend_handles = [
            Line2D([0], [0], marker="o", color="#444", lw=1.2,
                   markersize=4.5, markeredgecolor="white",
                   label="observed", linestyle="-"),
            Line2D([0], [0], marker="o", color="#444", lw=0,
                   markersize=4.5, markerfacecolor="white",
                   markeredgecolor="#444", markeredgewidth=1.0,
                   label="expected (additive)"),
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper right", bbox_to_anchor=(1.0, 1.04),
            fontsize=6, ncol=1,
            handlelength=1.2, handleheight=0.9,
            borderaxespad=0.2,
        )
        fig.tight_layout()

    return fig, ax
