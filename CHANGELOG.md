# Changelog

## 3.2.0 — 2026-05-15

### Classifier redesign

- **3-bucket classification** anchored on direct (d1, d2, de) comparisons rather than the additive residual: `rescue` · `synergistic` · `compounding`. Dropped the `antagonistic` bucket — most of its calls were saturation artifacts.
- Five sub-rules under `synergistic`: super-additive, emergent-at-edge (ref pegged near 0 or 1 — logit-scale meaningful), flip-direction (joint produces opposite effect of singles), with explicit guards against "joint matches the opposing single" dominance cases.
- `rescue` requires substantial worst single, joint near WT, real reduction, and same side of WT (with tiny-overshoot tolerance).
- `compounding` requires **both** singles to contribute substantially (min |d| ≥ 0.20) — small-residual events driven by a single dominant mutation no longer get a compounding label.

### Output

- `scan()` now optionally writes a **per-single missplicing CSV** alongside the per-pair CSV. One row per unique mutation with `max_abs_delta`, `max_delta_position`, `max_delta_site_type`, `max_delta_annotated`, missplicing flag, and counts at multiple Δ thresholds.

### Performance

- `_assemble_site_table` and `compute_site_residuals_multi` vectorized — **~23× speedup** on real gene scans. Replaces `iterrows` + per-row `pd.Series` construction with numpy boolean masks and `pd.DataFrame.melt`.
- Bulk classification of an 800K-pair TCGA dataset now finishes in ~22 hours on an L40S; previously ~8 days.

### Correctness fixes

- `_assemble_site_table` uses **union** of positions across contexts (not intersection). Previously, a deletion-induced gap in mut1's index would drop the canonical splice site from the table entirely. Now missing positions get prob=0.0 in that context — the canonical site stays visible and gets classified correctly (e.g. CREBBP A 3,758,048 compounding loss).
- `_select_transcript` overridden to walk all transcripts by their (start, end) bounds, recovering ~10% of genes whose primary isoform doesn't contain the variant position but other isoforms do (ABL1, ABL2, ABCA1, etc.).
- Classifier preserves KRAS-style canonical rescue when the joint is at WT within float-tolerance (tiny overshoots past WT no longer disqualify rescue).

### API

- `scan(constructs, ..., singles_checkpoint_path=...)` — optional per-single CSV path.
- `splicing_outcome_fingerprint()` + `splicing_outcome_hash()` for grouping events by gross splicing-impact pattern (logit-style discretization).

### Dependency simplification

- **`geney` is now optional.** The classification core (`analyze_pair`, `scan`, `classify_dataframe`) runs with only `numpy`, `pandas`, `matplotlib`, `biopython`, `seqmat`. The protein-library / Oncosplice protein-divergence path moves to the new `[protein]` extra (`pip install oncosplice[protein]`).
- `_apply_mutation_safe` inlined from geney (small helper that tolerates ref/alt mismatches on top of seqmat's `apply_mutations`).
- `select_transcript` is now a local implementation in `_geney_compat.py` — geney's own version had a bug that silently dropped ~10% of genes.

### Tooling

- GitHub Actions CI on Python 3.10 / 3.11 / 3.12.
- MIT LICENSE, CHANGELOG, two example notebooks (KRAS rescue, CREBBP synergistic).

## 3.1.0

- Switched production splicing engine from `spliceai_keras` to `spliceai_pytorch` (numerically identical, no TF dependency, ~2.5× faster).
- 5 engine adapters under one uniform interface: OpenSpliceAI, SpliceAI-PyTorch, Pangolin, Spliceformer, SpliceAI-Keras.

## 3.0.0

- First version with multi-engine support and the residual-based classifier.
