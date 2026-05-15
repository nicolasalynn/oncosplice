"""OncospliceEngine — single entry point for splicing-epistasis analysis.

The engine wires together:

- a :class:`SplicingPredictor` from :mod:`oncosplice.engines` (SpliceAI-Keras,
  OpenSpliceAI, Pangolin, Spliceformer, or any ensemble)
- ``seqmat`` for gene / transcript / pre-mRNA handling
- ``geney``'s ``SpliceSimulator`` for alternative-isoform enumeration (the
  protein library step from the Oncosplice paper)
- the scoring primitives in :mod:`oncosplice.scoring`

Public API:

- :meth:`analyze_single`  — one mutation → :class:`SingleVariantResult`
- :meth:`analyze_pair`    — two mutations → :class:`DoubleVariantResult`
- :meth:`analyze_multi`   — N mutations → :class:`MultiVariantResult`
- :meth:`scan`            — batched residual scan over many constructs
- :meth:`score_pairs_dataframe` — convenience wrapper for analyse_pair on a DataFrame

The legacy 3.0.0 names ``scan_pairs_residuals*`` / ``scan_multi_residuals*`` were
consolidated into :meth:`scan` (single dispatch on construct size).
"""
from __future__ import annotations

import csv as _csv
import logging
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .engines import SplicingPredictor, get_predictor
from .results import (
    DoubleVariantResult,
    MissplicingProfile,
    MultiVariantResult,
    SingleVariantResult,
)
from .scoring.epistasis import (
    DEFAULT_RESIDUAL_THRESHOLD,
    compute_site_residuals_multi,
    summarize_residuals,
)
from .scoring.oncosplice import aggregate_isoform_scores, oncosplice_score
from .scoring.splicing import (
    DEFAULT_DELTA_THRESHOLD,
    classify_missplicing,
    max_splicing_delta,
    missplicing_to_dict,
)
from .variants import Variant, VariantPair

logger = logging.getLogger(__name__)

SPLICING_CONTEXT_BP = 7500  # half-window for the local pre-mRNA region


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class OncospliceEngine:
    """Stateful entry point for single- / double- / N-variant analysis.

    Parameters
    ----------
    splicing_engine
        Name (or :class:`SplicingPredictor` instance) for splice-site
        prediction. Accepted names: ``"openspliceai"`` (default),
        ``"spliceai_keras"``, ``"pangolin"``, ``"spliceformer"``,
        ``"average"``, ``"ensemble:a,b,c"``. See :mod:`oncosplice.engines`.
    organism
        Genome build for ``seqmat`` (default ``"hg38"``).
    device
        Override device hint forwarded to the predictor.
    delta_threshold
        Per-site Δprob threshold for missplicing classification (paper default 0.25).
    residual_threshold
        |residual| threshold for site-level epistasis classification (paper default 0.25).
    gene_cache_size
        LRU cache size for ``seqmat.Gene`` objects.
    """

    def __init__(
        self,
        splicing_engine: Union[str, SplicingPredictor] = "openspliceai",
        organism: str = "hg38",
        device: Optional[str] = None,
        delta_threshold: float = DEFAULT_DELTA_THRESHOLD,
        residual_threshold: float = DEFAULT_RESIDUAL_THRESHOLD,
        gene_cache_size: int = 32,
    ):
        if isinstance(splicing_engine, SplicingPredictor):
            self.predictor: SplicingPredictor = splicing_engine
        else:
            kwargs = {"device": device} if device is not None else {}
            self.predictor = get_predictor(splicing_engine, **kwargs)
        self.splicing_engine = self.predictor.name
        self.organism = organism
        self.delta_threshold = float(delta_threshold)
        self.residual_threshold = float(residual_threshold)
        self._gene_cache: "OrderedDict[Tuple[str, str], object]" = OrderedDict()
        self._gene_cache_max = int(gene_cache_size)

    # ------------------------------------------------------------------
    # Reference-side helpers (gene / transcript / pre-mRNA)
    # ------------------------------------------------------------------
    def _load_gene(self, gene_name: str):
        from seqmat import Gene
        key = (gene_name, self.organism)
        if key in self._gene_cache:
            self._gene_cache.move_to_end(key)
            return self._gene_cache[key]
        gene = Gene.from_file(gene_name, organism=self.organism)
        if len(self._gene_cache) >= self._gene_cache_max:
            self._gene_cache.popitem(last=False)
        self._gene_cache[key] = gene
        return gene

    def _select_transcript(self, gene, central_pos: int, transcript_id: Optional[str]):
        from ._geney_compat import select_transcript
        return select_transcript(gene, central_pos, transcript_id)

    def _prepare_reference(self, gene_name: str, central_pos: int,
                           transcript_id: Optional[str], window_lo: int,
                           window_hi: int) -> Tuple[object, object]:
        """Load gene, pick transcript, generate mature mRNA + protein + pre-mRNA window.

        Returns ``(ref_transcript, ref_pre_mrna_transcript)`` — two clones; the
        first carries protein/mature_mrna, the second has a pre_mRNA over the
        requested window.
        """
        gene = self._load_gene(gene_name)
        ref = self._select_transcript(gene, central_pos, transcript_id)
        if ref is None:
            raise ValueError(
                f"No transcript of {gene_name} contains position {central_pos}."
            )
        ref = ref.generate_mature_mrna().generate_protein()
        ref_pre = ref.clone()
        ref_pre.generate_pre_mrna(region_start=window_lo, region_end=window_hi)
        return ref, ref_pre

    # ------------------------------------------------------------------
    # Splicing prediction — uniform across all contexts and engines
    # ------------------------------------------------------------------
    def _make_padded_seq(self, ref_pre, center: int, mutations: Sequence[Tuple[int, str, str]]
                        ) -> Tuple[str, np.ndarray]:
        """Apply mutations to a pre-mRNA clone, then build a padded ACGTN
        string and the genomic-position index for the biological (un-padded)
        region the predictor will return values for.

        The output sequence is ``2 * full_half + 1`` bp wide, where
        ``full_half = cl + analysis_half`` — ``cl`` bp of context that the
        predictor consumes plus ``analysis_half`` bp of biological region we
        actually want predictions for on each side. Returns
        ``(padded_sequence, biological_indices)`` with
        ``len(biological_indices) == len(padded_sequence) - 2*cl``.
        """
        try:
            from geney.utils import _apply_mutation_safe
        except ImportError:
            from geney.transcripts import _apply_mutation_safe  # type: ignore

        cl = self.predictor.context_length
        # How wide a biological window to predict on each side of `center`.
        # Match the legacy oncosplice 3.0.0 behaviour: ±(SPLICING_CONTEXT_BP − cl)
        # for short-context models (SpliceAI/Pangolin: ±2500); for long-context
        # models (Spliceformer: cl=20000) shrink to a sensible default.
        analysis_half = max(SPLICING_CONTEXT_BP - cl, 250)
        full_half = cl + analysis_half

        t = ref_pre.clone()
        t.generate_pre_mrna(region_start=center - full_half,
                            region_end=center + full_half)
        for pos, ref, alt in mutations:
            _apply_mutation_safe(t.pre_mrna, pos, ref, alt, permissive=True)

        pm = t.pre_mrna
        target = pm.clone(center - full_half, center + full_half)
        seq, indices = target.seq, target.index

        # N-pad if we're at a chromosome edge.
        rel_pos = int(np.abs(indices - center).argmin())
        left_missing  = max(0, full_half - rel_pos)
        right_missing = max(0, full_half - (len(seq) - rel_pos))
        if left_missing or right_missing:
            step = -1 if pm.rev else 1
            left_pad = (np.arange(indices[0] - step * left_missing, indices[0], step)
                        if left_missing else np.array([], dtype=indices.dtype))
            right_pad = (np.arange(indices[-1] + step,
                                   indices[-1] + step * (right_missing + 1), step)
                         if right_missing else np.array([], dtype=indices.dtype))
            seq = "N" * left_missing + seq + "N" * right_missing
            indices = np.concatenate([left_pad, indices, right_pad])

        # Predictor output corresponds to indices[cl:-cl] (drops the context
        # that the predictor consumed on each side of the input).
        biological_indices = indices[cl:-cl] if cl else indices
        return seq.upper(), biological_indices

    def _predict_context(self, ref_pre, center: int, mutations: Sequence[Tuple[int, str, str]]
                         ) -> Tuple[pd.Series, pd.Series]:
        """Run the predictor for one context. Returns donor & acceptor Series
        indexed by genomic position.
        """
        seq, idx = self._make_padded_seq(ref_pre, center, mutations)
        pred = self.predictor.predict(seq)
        if pred.length != len(idx):
            raise RuntimeError(
                f"predictor output length ({pred.length}) does not match index "
                f"length ({len(idx)}) — engine={self.splicing_engine!r}"
            )
        donor    = pd.Series(pred.donor,    index=idx)
        acceptor = pd.Series(pred.acceptor, index=idx)
        # Drop duplicate positions (windows may overlap at gene-edge padding).
        donor    = donor[~donor.index.duplicated(keep="first")]
        acceptor = acceptor[~acceptor.index.duplicated(keep="first")]
        return donor, acceptor

    # ------------------------------------------------------------------
    # site_table assembly (long format) — uniform across pair / multi
    # ------------------------------------------------------------------
    @staticmethod
    def _assemble_site_table(
        context_predictions: "dict[str, tuple[pd.Series, pd.Series]]",
        annotated_donors:    set,
        annotated_acceptors: set,
    ) -> pd.DataFrame:
        """Build the canonical long-format site_table from per-context predictions.

        ``context_predictions``: ordered dict ``{ctx_name: (donor_series, acceptor_series)}``.
        Returns a DataFrame with columns [position, site_type, context, prob, annotated].
        """
        # Intersect all indices so every context has a value at every reported site.
        ctxs = list(context_predictions.keys())
        if not ctxs:
            return pd.DataFrame(columns=["position", "site_type", "context", "prob", "annotated"])

        # Vectorized assembly: build one wide frame per site_type with
        # contexts as columns, then melt → long. Avoids the per-(pos,ctx) dict
        # construction that dominated `scan()` runtime on real genes.
        frames = []
        ctx_names = list(context_predictions.keys())
        for site_type, ann_set in (("donor", annotated_donors),
                                    ("acceptor", annotated_acceptors)):
            # Union of positions across all contexts + the annotation set
            # (annotated sites must always appear — see CREBBP A 3,758,048
            # bug where a deletion-induced gap dropped the canonical site).
            all_positions = set()
            for _ctx, (d, a) in context_predictions.items():
                s = d if site_type == "donor" else a
                all_positions.update(s.index.to_numpy().astype(int).tolist())
            all_positions.update(int(p) for p in ann_set)
            if not all_positions:
                continue
            positions = sorted(all_positions)
            idx = pd.Index(positions, dtype="int64")

            # Wide frame: one column per context, missing positions = 0.0
            cols = {"position": positions}
            for ctx, (d, a) in context_predictions.items():
                s = d if site_type == "donor" else a
                cols[ctx] = s.reindex(idx, fill_value=0.0).to_numpy(dtype=float)
            wide = pd.DataFrame(cols)
            wide["site_type"] = site_type
            wide["annotated"] = wide["position"].isin(ann_set)

            # Melt to long format expected downstream.
            long = wide.melt(
                id_vars=["position", "site_type", "annotated"],
                value_vars=ctx_names,
                var_name="context",
                value_name="prob",
            )
            frames.append(long)

        if not frames:
            return pd.DataFrame(columns=["position", "site_type", "context", "prob", "annotated"])
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _annotated_sets(ref_transcript) -> Tuple[set, set]:
        d = getattr(ref_transcript, "donors", None)
        a = getattr(ref_transcript, "acceptors", None)
        return (set(int(x) for x in (d if d is not None else [])),
                set(int(x) for x in (a if a is not None else [])))

    # ------------------------------------------------------------------
    # Isoform scoring (protein library)
    # ------------------------------------------------------------------
    def _score_isoforms(
        self,
        ref_transcript,
        site_table_for_context: pd.DataFrame,
        context: str,
        max_isoforms: Optional[int],
    ) -> Tuple[pd.DataFrame, float, float]:
        """Enumerate alternative isoforms via SpliceSimulator and score each.

        Returns (isoforms_df, aggregate_score, max_percentile).
        """
        from ._geney_compat import SpliceSimulator

        # SpliceSimulator wants the wide multi-index format produced by
        # geney's adjoin_splicing_outcomes. We materialize a small wide
        # frame from our long site_table so the simulator can run.
        wide_for_ctx = self._site_table_to_legacy_wide(site_table_for_context, context)
        # TranscriptLibrary wraps the wide frame and exposes the simulator API.
        # We construct an empty event for the wrapper; the wide frame already
        # carries the prediction data the simulator needs.
        target_transcript = ref_transcript  # ref + mutations applied by SS internally? — no, SS just walks the splice graph
        ss = SpliceSimulator(
            wide_for_ctx,
            target_transcript,
            feature=context,
            max_distance=100_000_000,
        )
        cons_vector = getattr(ref_transcript, "cons_vector", None)
        if cons_vector is None:
            cons_vector = np.ones(len(getattr(ref_transcript, "protein", "") or "X"))
        cons_vector = np.asarray(cons_vector, dtype=float)

        try:
            isoforms_iter = ss.get_viable_transcripts(metadata=True, max_isoforms=max_isoforms)
        except TypeError:
            isoforms_iter = ss.get_viable_transcripts(metadata=True)

        rows = []
        score_x_prev: list[Tuple[float, float]] = []
        for n, (var_t, md) in enumerate(isoforms_iter):
            if max_isoforms is not None and n >= max_isoforms:
                break
            score = oncosplice_score(
                ref_transcript.protein or "",
                var_t.protein or "",
                cons_vector,
            )
            row = {
                "isoform_id":       md.get("isoform_id", ""),
                "prevalence":       float(md.get("isoform_prevalence", 0.0)),
                "splicing_changes": md.get("summary", "-"),
                "oncosplice_score": score.score,
                "percentile":       score.percentile,
                "n_deletions":      score.n_deletions,
                "n_insertions":     score.n_insertions,
                "variant_protein":  var_t.protein or "",
                "es":  md.get("es", ""),
                "ir":  md.get("ir", ""),
                "pes": md.get("pes", ""),
                "pir": md.get("pir", ""),
                "ne":  md.get("ne", ""),
            }
            rows.append(row)
            score_x_prev.append((score.score, row["prevalence"]))

        df = pd.DataFrame(rows)
        agg_score = aggregate_isoform_scores(score_x_prev)
        agg_pctl = float(df["percentile"].max()) if not df.empty else 0.0
        return df, agg_score, agg_pctl

    @staticmethod
    def _site_table_to_legacy_wide(site_table: pd.DataFrame, context: str) -> pd.DataFrame:
        """Adapt our long-format site_table back to geney's wide multi-index format
        for ``SpliceSimulator`` compatibility. Returns a DataFrame indexed by
        position with MultiIndex columns ``(site_type, metric)``.
        """
        if site_table.empty:
            return site_table
        wide = (
            site_table.assign(metric=lambda d: d.context.map(
                lambda c: "ref_prob" if c == "ref" else f"{c}_prob"))
            .pivot_table(
                index=["position", "site_type", "annotated"],
                columns="metric",
                values="prob",
                aggfunc="first",
            )
            .reset_index()
        )
        wide.columns.name = None
        # Now reshape into multi-index columns expected by geney's downstream code
        rows = []
        for st in ("donor", "acceptor"):
            sub = wide[wide.site_type == st].copy()
            if sub.empty: continue
            sub = sub.set_index("position")
            sub.columns = pd.MultiIndex.from_product([[f"{st}s"], sub.columns])
            rows.append(sub)
        if not rows:
            return pd.DataFrame()
        out = pd.concat(rows, axis=1)
        return out

    # ------------------------------------------------------------------
    # Public: single-variant analysis
    # ------------------------------------------------------------------
    def analyze_single(
        self,
        mut_id: str,
        *,
        transcript_id: Optional[str] = None,
        max_isoforms: Optional[int] = None,
        protein: bool = True,
    ) -> SingleVariantResult:
        v = Variant.from_id(mut_id)
        center = v.pos
        cl = self.predictor.context_length
        window = max(cl, SPLICING_CONTEXT_BP)
        ref, ref_pre = self._prepare_reference(
            v.gene, center, transcript_id,
            window_lo=center - window, window_hi=center + window,
        )
        ann_d, ann_a = self._annotated_sets(ref)

        # Predict ref + event contexts
        ref_d,  ref_a  = self._predict_context(ref_pre, center, mutations=[])
        ev_d,   ev_a   = self._predict_context(ref_pre, center, mutations=[(v.pos, v.ref, v.alt)])

        site_table = self._assemble_site_table(
            {"ref": (ref_d, ref_a), "event": (ev_d, ev_a)},
            ann_d, ann_a,
        )

        events = classify_missplicing(site_table, context="event", threshold=self.delta_threshold)
        missplicing = MissplicingProfile(**missplicing_to_dict(events))

        if protein:
            isoforms_df, agg_score, agg_pctl = self._score_isoforms(
                ref, site_table, context="event", max_isoforms=max_isoforms,
            )
        else:
            isoforms_df = pd.DataFrame()
            agg_score = 0.0
            agg_pctl = 0.0

        return SingleVariantResult(
            mut_id=v.mut_id,
            gene=v.gene,
            transcript_id=getattr(ref, "transcript_id", ""),
            splicing_engine=self.splicing_engine,
            central_position=int(center),
            missplicing=missplicing,
            max_splicing_delta=max_splicing_delta(site_table, context="event"),
            isoforms=isoforms_df,
            oncosplice_score=float(agg_score),
            percentile=float(agg_pctl),
            site_table=site_table,
            reference_protein=ref.protein or "",
            reference_mrna=getattr(ref.mature_mrna, "seq", "") or "",
        )

    # ------------------------------------------------------------------
    # Public: double-variant analysis (thin wrapper around analyze_multi)
    # ------------------------------------------------------------------
    def analyze_pair(
        self,
        mut1_id: str,
        mut2_id: Optional[str] = None,
        *,
        transcript_id: Optional[str] = None,
        max_isoforms: Optional[int] = None,
        protein: bool = True,
    ) -> DoubleVariantResult:
        pair = VariantPair.from_ids(mut1_id, mut2_id)
        m = self.analyze_multi(
            [pair.mut1.mut_id, pair.mut2.mut_id],
            transcript_id=transcript_id,
            max_isoforms=max_isoforms,
            protein=protein,
        )
        # Repackage the multi result as a typed DoubleVariantResult.
        return DoubleVariantResult(
            mut1_id=pair.mut1.mut_id,
            mut2_id=pair.mut2.mut_id,
            epistasis_id=pair.epistasis_id,
            gene=pair.gene,
            transcript_id=m.transcript_id,
            splicing_engine=self.splicing_engine,
            distance=pair.distance,
            central_position=int(m.central_position),
            pair_classification=m.pair_classification,
            score_residual=float(
                (m.oncosplice_score_event or 0.0)
                - ((m._mut_scores or [0.0, 0.0])[0] + (m._mut_scores or [0.0, 0.0])[1] - 0.0)
            ) if getattr(m, "_mut_scores", None) else 0.0,
            oncosplice_scores={
                "ref": 0.0,
                "mut1": (m._mut_scores or [0.0, 0.0])[0] if getattr(m, "_mut_scores", None) else 0.0,
                "mut2": (m._mut_scores or [0.0, 0.0])[1] if getattr(m, "_mut_scores", None) else 0.0,
                "event": m.oncosplice_score_event or 0.0,
            },
            oncosplice_percentiles={},
            site_residuals=m.site_residuals,
            epistasis_summary=m.epistasis_summary,
            site_table=m.site_table,
            isoforms_event=m.isoforms_event,
            isoforms_mut1=(m._mut_isoforms or [None, None])[0] if getattr(m, "_mut_isoforms", None) else None,
            isoforms_mut2=(m._mut_isoforms or [None, None])[1] if getattr(m, "_mut_isoforms", None) else None,
            reference_protein=m.reference_protein,
        )

    # ------------------------------------------------------------------
    # Public: N-variant analysis (the canonical implementation)
    # ------------------------------------------------------------------
    def analyze_multi(
        self,
        mut_ids: Union[Sequence[str], str],
        *,
        transcript_id: Optional[str] = None,
        max_isoforms: Optional[int] = None,
        protein: bool = True,
    ) -> MultiVariantResult:
        """Splicing residuals + classification for an N-variant construct.

        For N ≥ 2 we predict ``ref`` + each individual mutant + the joint event;
        residuals are ``event − (sum(mut_i) − (N−1)·ref)``.

        When ``protein=True`` (default) the joint event also gets a protein
        library via SpliceSimulator; constituent-protein scores are also
        computed so :meth:`analyze_pair` can derive ``score_residual``.
        """
        if isinstance(mut_ids, str):
            mut_ids = [m.strip() for m in mut_ids.split("|") if m.strip()]
        if len(mut_ids) < 2:
            raise ValueError("analyze_multi requires at least 2 mutations.")
        variants = [Variant.from_id(m) for m in mut_ids]
        gene = variants[0].gene
        if any(v.gene != gene for v in variants):
            raise ValueError(f"All variants must share a gene; got {[v.gene for v in variants]}.")
        canonical = [v.mut_id for v in variants]
        construct_id = "|".join(canonical)
        n = len(variants)

        center = (min(v.pos for v in variants) + max(v.pos for v in variants)) // 2
        cl = self.predictor.context_length
        window = max(cl, SPLICING_CONTEXT_BP)
        ref, ref_pre = self._prepare_reference(
            gene, center, transcript_id,
            window_lo=center - window, window_hi=center + window,
        )
        ann_d, ann_a = self._annotated_sets(ref)

        # Predictions: ref + each single + event
        context_preds: "dict[str, tuple[pd.Series, pd.Series]]" = {}
        context_preds["ref"] = self._predict_context(ref_pre, center, mutations=[])
        for i, v in enumerate(variants, start=1):
            context_preds[f"mut{i}"] = self._predict_context(
                ref_pre, center, mutations=[(v.pos, v.ref, v.alt)],
            )
        context_preds["event"] = self._predict_context(
            ref_pre, center, mutations=[(v.pos, v.ref, v.alt) for v in variants],
        )

        site_table = self._assemble_site_table(context_preds, ann_d, ann_a)
        site_residuals = compute_site_residuals_multi(
            site_table, n_variants=n, threshold=self.residual_threshold,
        )
        summary = summarize_residuals(site_residuals)

        # Protein library (joint context + optional per-single)
        isoforms_event = None
        agg_event = None
        mut_scores: Optional[List[float]] = None
        mut_isoforms: Optional[List[pd.DataFrame]] = None
        if protein:
            isoforms_event, agg_event, _ = self._score_isoforms(
                ref, site_table, context="event", max_isoforms=max_isoforms,
            )
            mut_scores = []
            mut_isoforms = []
            for i in range(1, n + 1):
                df_i, agg_i, _ = self._score_isoforms(
                    ref, site_table, context=f"mut{i}", max_isoforms=max_isoforms,
                )
                mut_scores.append(float(agg_i))
                mut_isoforms.append(df_i)

        result = MultiVariantResult(
            construct_id=construct_id,
            mut_ids=canonical,
            gene=gene,
            transcript_id=getattr(ref, "transcript_id", ""),
            splicing_engine=self.splicing_engine,
            n_variants=n,
            central_position=int(center),
            pair_classification=summary["pair_classification"],
            site_residuals=site_residuals,
            epistasis_summary=summary,
            site_table=site_table,
            isoforms_event=isoforms_event,
            oncosplice_score_event=agg_event,
            reference_protein=ref.protein or "",
        )
        # Stash auxiliary scores for analyze_pair's repackaging step.
        # Use private attrs so they're not part of the public dataclass API.
        result._mut_scores   = mut_scores       # type: ignore[attr-defined]
        result._mut_isoforms = mut_isoforms     # type: ignore[attr-defined]
        return result

    # ------------------------------------------------------------------
    # Public: batched scan (the only scan method; replaces 4 legacy ones)
    # ------------------------------------------------------------------
    def scan(
        self,
        constructs: Union[Sequence[str], Sequence[Sequence[str]]],
        *,
        transcript_id: Optional[str] = None,
        progress: bool = True,
        checkpoint_path: Optional[Union[str, Path]] = None,
        singles_checkpoint_path: Optional[Union[str, Path]] = None,
        resume: bool = True,
        batch_size: int = 32,
    ) -> pd.DataFrame:
        """Fast residual-only scan over many constructs (each is a list / pipe-string of mutations).

        Strategy:
        1. Enumerate every *unique* mutated pre-mRNA sequence (ref + singles + joints).
        2. Run all of them through :meth:`SplicingPredictor.predict_many` —
           predictors with batched inference (OpenSpliceAI on GPU/MPS) get a
           ~10–50× speedup; predictors with only a per-sequence path still
           benefit from caching across constructs.
        3. Stitch per-construct site tables and compute residuals.

        Optional ``checkpoint_path`` writes each completed row to CSV
        immediately for resume-friendly long jobs.
        """
        # ---- Normalize ---------------------------------------------------
        normalized: list[list[str]] = []
        for c in constructs:
            if isinstance(c, str):
                normalized.append([m.strip() for m in c.split("|") if m.strip()])
            else:
                normalized.append([m.strip() for m in c])
        if not normalized:
            return pd.DataFrame()

        all_vs = [[Variant.from_id(m) for m in cs] for cs in normalized]
        gene = next((vs[0].gene for vs in all_vs if vs), None)
        if gene is None:
            return pd.DataFrame()
        if not all(all(v.gene == gene for v in vs) for vs in all_vs):
            raise ValueError("All constructs must share a gene (run one scan per gene).")
        all_positions = [v.pos for vs in all_vs for v in vs]
        center = (min(all_positions) + max(all_positions)) // 2
        cl = self.predictor.context_length
        window = max(cl, SPLICING_CONTEXT_BP) + (max(all_positions) - min(all_positions))
        ref, ref_pre = self._prepare_reference(
            gene, center, transcript_id,
            window_lo=center - window, window_hi=center + window,
        )
        ann_d, ann_a = self._annotated_sets(ref)

        # ---- Resume support ---------------------------------------------
        cp_path = Path(checkpoint_path) if checkpoint_path else None
        completed: set[str] = set()
        if cp_path is not None and cp_path.exists() and resume:
            try:
                prior = pd.read_csv(cp_path)
                completed = set(prior.get("construct_id", pd.Series(dtype=str)).astype(str))
                logger.info("resume: %d already in %s", len(completed), cp_path.name)
            except Exception:
                completed = set()

        # ---- Enumerate unique sequences ---------------------------------
        single_keys: set[Tuple[int, str, str]] = set()
        single_mut_id: dict[Tuple[int, str, str], str] = {}
        joint_keys: list[Tuple[Tuple[int, str, str], ...]] = []
        joint_idx: dict = {}
        construct_keys: list[Optional[Tuple[Tuple[int, str, str], ...]]] = []
        for vs in all_vs:
            cid = "|".join(v.mut_id for v in vs)
            if cid in completed:
                construct_keys.append(None)
                continue
            for v in vs:
                key = (v.pos, v.ref, v.alt)
                single_keys.add(key)
                single_mut_id.setdefault(key, v.mut_id)
            jk = tuple((v.pos, v.ref, v.alt) for v in vs)
            construct_keys.append(jk)
            if len(vs) >= 2 and jk not in joint_idx:
                joint_idx[jk] = len(joint_keys)
                joint_keys.append(jk)
        if not single_keys and not joint_keys:
            return pd.DataFrame()

        # Build the padded sequence string for each unique context.
        seq_index: dict = {}  # key → row index in the sequences list
        seqs: list[str] = []
        idxs: list[np.ndarray] = []

        def _add(mutations: Tuple[Tuple[int, str, str], ...], key):
            seq, idx = self._make_padded_seq(ref_pre, center, list(mutations))
            seq_index[key] = len(seqs)
            seqs.append(seq); idxs.append(idx)

        _add((), "_ref")
        for sk in single_keys:
            _add((sk,), ("single", sk))
        for jk in joint_keys:
            _add(jk, ("joint", jk))

        # ---- Run predictor.predict_many (batched if supported) ----------
        logger.info("scan: %d unique sequences (1 ref + %d singles + %d joints)",
                    len(seqs), len(single_keys), len(joint_keys))
        preds = self.predictor.predict_many(seqs)

        def _series_pair(key) -> Tuple[pd.Series, pd.Series]:
            i = seq_index[key]
            idx = idxs[i]
            d = pd.Series(preds[i].donor, index=idx)
            a = pd.Series(preds[i].acceptor, index=idx)
            return d[~d.index.duplicated(keep="first")], a[~a.index.duplicated(keep="first")]

        ref_d, ref_a = _series_pair("_ref")

        # ---- Per-single missplicing summary (optional) ------------------
        # If the caller passed ``singles_checkpoint_path``, emit a one-row-per-
        # unique-single summary alongside the pair output. Re-uses the already-
        # computed single predictions; no extra inference cost.
        sing_cp_path = Path(singles_checkpoint_path) if singles_checkpoint_path else None
        if sing_cp_path is not None and single_keys:
            SINGLES_FIELDS = [
                "mut_id", "gene",
                "max_abs_delta", "max_delta_signed",
                "max_delta_position", "max_delta_site_type",
                "max_delta_annotated", "max_mut_prob", "max_ref_prob",
                "n_sites_delta_0.15", "n_sites_delta_0.30",
                "missplicing", "engine", "error",
            ]
            # Resume: skip singles already summarised.
            sing_done: set[str] = set()
            if sing_cp_path.exists() and resume:
                try:
                    prior_s = pd.read_csv(sing_cp_path)
                    sing_done = set(prior_s.get("mut_id", pd.Series(dtype=str)).astype(str))
                except Exception:
                    sing_done = set()

            sing_fh = open(sing_cp_path, "a", newline="")
            try:
                writer = _csv.DictWriter(sing_fh, fieldnames=SINGLES_FIELDS, extrasaction="ignore")
                if sing_cp_path.stat().st_size == 0:
                    writer.writeheader()
                for sk in single_keys:
                    mid = single_mut_id.get(sk, f"{gene}:{sk[0]}:{sk[1]}:{sk[2]}")
                    if mid in sing_done:
                        continue
                    try:
                        m_d, m_a = _series_pair(("single", sk))
                        # Build long arrays of (position, site_type, ref_prob, mut_prob, annotated)
                        # across union of indices, fill-missing with 0 (matches _assemble_site_table).
                        all_d_pos = sorted(set(ref_d.index.astype(int)) | set(m_d.index.astype(int)) | {p for p in ann_d})
                        all_a_pos = sorted(set(ref_a.index.astype(int)) | set(m_a.index.astype(int)) | {p for p in ann_a})
                        rows_arr = []
                        if all_d_pos:
                            idx = pd.Index(all_d_pos, dtype="int64")
                            r_ = ref_d.reindex(idx, fill_value=0.0).to_numpy(dtype=float)
                            m_ = m_d.reindex(idx, fill_value=0.0).to_numpy(dtype=float)
                            rows_arr.append(("donor", all_d_pos, r_, m_, ann_d))
                        if all_a_pos:
                            idx = pd.Index(all_a_pos, dtype="int64")
                            r_ = ref_a.reindex(idx, fill_value=0.0).to_numpy(dtype=float)
                            m_ = m_a.reindex(idx, fill_value=0.0).to_numpy(dtype=float)
                            rows_arr.append(("acceptor", all_a_pos, r_, m_, ann_a))

                        best = {"abs": -1.0, "delta": 0.0, "pos": 0, "site_type": "",
                                "ann": False, "mut": 0.0, "ref": 0.0}
                        n_15 = 0; n_30 = 0
                        for site_type, positions, r_, m_, ann_set in rows_arr:
                            delta = m_ - r_
                            abs_d = np.abs(delta)
                            n_15 += int((abs_d >= 0.15).sum())
                            n_30 += int((abs_d >= 0.30).sum())
                            if len(abs_d) == 0:
                                continue
                            i_max = int(abs_d.argmax())
                            if abs_d[i_max] > best["abs"]:
                                best = {
                                    "abs": float(abs_d[i_max]),
                                    "delta": float(delta[i_max]),
                                    "pos": int(positions[i_max]),
                                    "site_type": site_type,
                                    "ann": int(positions[i_max]) in ann_set,
                                    "mut": float(m_[i_max]),
                                    "ref": float(r_[i_max]),
                                }
                        writer.writerow({
                            "mut_id":              mid,
                            "gene":                gene,
                            "max_abs_delta":       best["abs"] if best["abs"] >= 0 else 0.0,
                            "max_delta_signed":    best["delta"],
                            "max_delta_position":  best["pos"],
                            "max_delta_site_type": best["site_type"],
                            "max_delta_annotated": best["ann"],
                            "max_mut_prob":        best["mut"],
                            "max_ref_prob":        best["ref"],
                            "n_sites_delta_0.15":  n_15,
                            "n_sites_delta_0.30":  n_30,
                            "missplicing":         bool(best["abs"] >= 0.25),
                            "engine":              self.splicing_engine,
                            "error":               None,
                        })
                    except Exception as exc:  # noqa: BLE001
                        writer.writerow({"mut_id": mid, "gene": gene,
                                         "engine": self.splicing_engine, "error": str(exc)})
                sing_fh.flush()
            finally:
                sing_fh.close()

        # ---- Stitch per-construct residuals -----------------------------
        iterator = list(zip(normalized, all_vs, construct_keys))
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc=f"scan ({self.splicing_engine})")
            except ImportError:
                pass

        out_rows: list[dict] = []
        cp_writer = None; cp_fh = None
        FIELDS = ["construct_id", "n_variants", "pair_classification",
                  "max_abs_residual", "max_abs_event_delta",
                  "n_syn", "n_rescue", "n_compound", "n_ant",
                  "max_synergy_residual", "max_antagonism_residual",
                  "engine", "error"]
        try:
            for mut_id_list, vs, jk in iterator:
                if jk is None:
                    continue
                cid = "|".join(v.mut_id for v in vs)
                n = len(vs)
                try:
                    ctx_preds = {"ref": (ref_d, ref_a)}
                    for i, v in enumerate(vs, start=1):
                        ctx_preds[f"mut{i}"] = _series_pair(("single", (v.pos, v.ref, v.alt)))
                    ctx_preds["event"] = (_series_pair(("joint", jk)) if n >= 2
                                          else _series_pair(("single", (vs[0].pos, vs[0].ref, vs[0].alt))))
                    site_table = self._assemble_site_table(ctx_preds, ann_d, ann_a)
                    if n >= 2:
                        sr = compute_site_residuals_multi(
                            site_table, n_variants=n, threshold=self.residual_threshold,
                        )
                        summary = summarize_residuals(sr)
                        out_rows.append({
                            "construct_id":           cid,
                            "n_variants":             n,
                            "pair_classification":    summary["pair_classification"],
                            "max_abs_residual":       summary["max_abs_residual"],
                            "max_abs_event_delta":    summary["max_abs_event_delta"],
                            "n_syn":                  summary["n_syn"],
                            "n_rescue":               summary["n_rescue"],
                            "n_compound":             summary["n_compound"],
                            "n_ant":                  summary["n_ant"],
                            "max_synergy_residual":    summary["max_synergy_residual"],
                            "max_antagonism_residual": summary["max_antagonism_residual"],
                            "engine":                  self.splicing_engine,
                            "error":                  None,
                        })
                    else:
                        delta = max_splicing_delta(site_table, context="event")
                        out_rows.append({
                            "construct_id":     cid,
                            "n_variants":       1,
                            "pair_classification": "singleton",
                            "max_abs_residual":    abs(float(delta)),
                            "max_abs_event_delta": abs(float(delta)),
                            "n_syn": 0, "n_rescue": 0, "n_compound": 0, "n_ant": 0,
                            "engine":              self.splicing_engine,
                            "error":               None,
                        })
                except Exception as exc:  # noqa: BLE001
                    out_rows.append({"construct_id": cid, "engine": self.splicing_engine, "error": str(exc)})

                if cp_path is not None:
                    if cp_writer is None:
                        new_file = not cp_path.exists() or cp_path.stat().st_size == 0
                        cp_fh = open(cp_path, "a", newline="")
                        cp_writer = _csv.DictWriter(cp_fh, fieldnames=FIELDS, extrasaction="ignore")
                        if new_file:
                            cp_writer.writeheader()
                    cp_writer.writerow(out_rows[-1])
                    cp_fh.flush()
        finally:
            if cp_fh is not None:
                cp_fh.close()
        return pd.DataFrame(out_rows)

    # ------------------------------------------------------------------
    # Convenience: apply analyze_pair to every row of a DataFrame
    # ------------------------------------------------------------------
    def score_pairs_dataframe(
        self,
        df: pd.DataFrame,
        *,
        mut1_col: str = "mut1_id",
        mut2_col: str = "mut2_id",
        progress: bool = False,
    ) -> pd.DataFrame:
        """Apply :meth:`analyze_pair` to every row; merge summary columns back.

        Slow per-pair path (includes Oncosplice protein scoring). Use
        :meth:`classify_dataframe` for fast bulk classification.
        """
        iterator = df.iterrows()
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, total=len(df))
            except ImportError:
                pass
        records = []
        for idx, row in iterator:
            try:
                r = self.analyze_pair(row[mut1_col], row[mut2_col])
                records.append({"_idx": idx, "error": None, **r.summary()})
            except Exception as exc:  # noqa: BLE001
                logger.warning("analyze_pair failed for row %s: %s", idx, exc)
                records.append({"_idx": idx, "error": str(exc)})
        out = pd.DataFrame(records).set_index("_idx")
        return df.join(out, how="left")

    # ------------------------------------------------------------------
    # Convenience: bulk-classify a DataFrame of epistasis IDs (fast scan path)
    # ------------------------------------------------------------------
    def classify_dataframe(
        self,
        df: pd.DataFrame,
        *,
        epistasis_id_col: Optional[str] = None,
        mut1_col: Optional[str] = None,
        mut2_col: Optional[str] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        resume: bool = True,
        progress: bool = True,
        sort_by_gene: bool = True,
    ) -> pd.DataFrame:
        """Bulk-classify a DataFrame of pair / N-variant IDs using the batched
        :meth:`scan` path. Returns the input DataFrame with new columns:

        - ``pair_classification``  (synergistic / rescue / compounding /
          antagonistic / non-epistatic / singleton)
        - ``max_abs_residual``      max |event − expected| across sites
        - ``max_abs_event_delta``   max |event − ref| across sites
        - ``n_syn``, ``n_rescue``, ``n_compound``, ``n_ant``  per-class site counts
        - ``max_synergy_residual``, ``max_antagonism_residual``
        - ``engine``  splicing engine used
        - ``error``   None if successful, else the exception message

        Pass either ``epistasis_id_col`` (single column with
        ``"GENE:CHR:POS:REF:ALT|GENE:CHR:POS:REF:ALT"``) or both ``mut1_col``
        and ``mut2_col``.

        ``sort_by_gene=True`` (default) processes pairs grouped by gene so
        the gene cache + reference-context predictions are reused across
        every pair within a gene. ~10–40× speedup on multi-pair-per-gene
        datasets (typical for TCGA).

        ``checkpoint_path`` writes incremental rows so the run survives
        crashes; on restart, ``resume=True`` skips already-completed pairs.
        """
        # 1. Build a normalised list of (input_index, epistasis_id, gene)
        if epistasis_id_col is not None:
            ids = df[epistasis_id_col].astype(str).tolist()
        elif mut1_col is not None and mut2_col is not None:
            ids = (df[mut1_col].astype(str) + "|" + df[mut2_col].astype(str)).tolist()
        else:
            raise ValueError(
                "Provide either `epistasis_id_col` or both `mut1_col` and `mut2_col`."
            )

        rows = []
        for idx, eid in zip(df.index, ids):
            try:
                gene = eid.split(":", 1)[0].split("|", 1)[0]
            except Exception:
                gene = ""
            rows.append({"_input_idx": idx, "construct_id": eid, "gene": gene})
        work = pd.DataFrame(rows)

        # 2. Group by gene, run scan() per gene
        if sort_by_gene:
            work = work.sort_values(["gene", "construct_id"]).reset_index(drop=True)

        per_gene_results: list[pd.DataFrame] = []
        gene_groups = work.groupby("gene", sort=False)
        iterator = gene_groups
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(list(gene_groups), total=gene_groups.ngroups,
                                desc=f"classify ({self.splicing_engine})")
            except ImportError:
                pass

        for gene, gene_work in iterator:
            constructs = gene_work["construct_id"].tolist()
            try:
                res = self.scan(
                    constructs,
                    progress=False,                # outer tqdm handles progress
                    checkpoint_path=checkpoint_path,
                    resume=resume,
                )
            except Exception as exc:               # whole-gene failure
                res = pd.DataFrame([
                    {"construct_id": c, "engine": self.splicing_engine, "error": str(exc)}
                    for c in constructs
                ])
            per_gene_results.append(res)

        scan_out = pd.concat(per_gene_results, ignore_index=True) if per_gene_results else pd.DataFrame()

        # 3. Re-join scan output back to the original DataFrame on construct_id
        merged = work.merge(scan_out, on="construct_id", how="left")
        merged = merged.set_index("_input_idx").reindex(df.index)
        merged = merged.drop(columns=["construct_id", "gene"], errors="ignore")
        return df.join(merged, how="left")

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"OncospliceEngine(engine={self.splicing_engine!r}, "
            f"organism={self.organism!r}, "
            f"delta_threshold={self.delta_threshold}, "
            f"residual_threshold={self.residual_threshold})"
        )
