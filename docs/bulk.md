# Bulk analysis

`classify_dataframe()` is the production path for TCGA-shaped epistasis tables.

```python
import pandas as pd
from oncosplice import OncospliceEngine

eng = OncospliceEngine(splicing_engine="spliceai_pytorch")

df = pd.read_csv("pairs.csv")  # column: epistasis_id "GENE:CHR:POS:REF:ALT|GENE:CHR:POS:REF:ALT"
out = eng.classify_dataframe(
    df,
    epistasis_id_col="epistasis_id",
    checkpoint_path="results.csv",
    singles_checkpoint_path="singles.csv",
)
```

## Output columns added

| Column | Meaning |
|---|---|
| `pair_classification` | `synergistic` / `rescue` / `compounding` / `non-epistatic` |
| `max_abs_residual` | Maximum \|residual\| over splice sites |
| `max_abs_event_delta` | Maximum \|de\| over splice sites |
| `n_syn`, `n_rescue`, `n_compound` | Per-site counts |
| `engine` | Engine identifier used for this row |
| `error` | Error string, or empty on success |

The optional per-single CSV adds one row per unique mutation with
`max_abs_delta`, `max_delta_position`, `max_delta_site_type`,
`max_delta_annotated`, the missplicing flag, and counts at multiple Δ
thresholds.

## How the runner is fast

`classify_dataframe()` calls `scan()` per gene under the hood:

1. **Group by gene.** All pairs in the same gene share a sequence context, so
   single-variant predictions can be cached and reused across pairs.
2. **Batch inference.** Within a gene, all required (ref / mut1 / mut2 /
   event) contexts are stacked into a single batch — the GPU sees one big
   forward pass instead of one per pair.
3. **Vectorized site assembly.** `_assemble_site_table` and
   `compute_site_residuals_multi` use NumPy boolean masks + `DataFrame.melt`
   rather than `iterrows`. **~23× faster** after the 3.2.0 vectorization.
4. **Resumable checkpointing.** Re-running with the same `checkpoint_path`
   skips already-done pairs and appends new results.

## Realistic numbers

On a single L40S GPU:

| Dataset | Pairs | Wall time |
|---|---|---|
| 800k TCGA pairs (`spliceai_pytorch`) | 800,000 | ~22 hours |
| 50k targeted pairs (`spliceai_pytorch`) | 50,000 | ~85 minutes |
| 800k pairs *pre-3.2.0* (per-`iterrows` site assembly) | 800,000 | ~8 days |

Engine choice dominates wall time. `pangolin` (40-model ensemble) is ~10×
slower than `spliceai_pytorch`; `spliceformer` (40k context) is similar to
`pangolin` due to long-context attention.

## Failure handling

A failure on one pair never aborts the run — the row gets an `error` string
and the runner continues. Common modes:

- **Out-of-gene variant** (mutation position outside the gene's annotated
  span) — recovered in 3.2.0 by walking *all* transcripts of the gene
  (`_select_transcript` overload), but still emitted as an error if no
  transcript covers the position.
- **Engine load failure** — typically a missing weight file. Run
  `oncosplice-download-weights --engine <name>` first.
- **Reference-allele mismatch** — the variant's REF doesn't match the
  reference sequence at that position. Logged with the expected base.
