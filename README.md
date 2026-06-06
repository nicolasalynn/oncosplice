# oncosplice

[![PyPI](https://img.shields.io/pypi/v/oncosplice.svg)](https://pypi.org/project/oncosplice/)
[![CI](https://github.com/nicolasalynn/oncosplice/actions/workflows/ci.yml/badge.svg)](https://github.com/nicolasalynn/oncosplice/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/nicolasalynn/oncosplice/branch/main/graph/badge.svg)](https://codecov.io/gh/nicolasalynn/oncosplice)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://nicolasalynn.github.io/oncosplice)
[![Python](https://img.shields.io/pypi/pyversions/oncosplice.svg)](https://pypi.org/project/oncosplice/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> Given two (or more) mutations in the same gene, classify how their joint
> effect on splicing differs from the additive prediction — into one of four
> mutually-exclusive mechanism classes: **rescue**, **cryptic rescue**,
> **deletion synergy**, or **cryptic synergy**.

**oncosplice** is a sequence-level pipeline for splicing-epistasis analysis of
single-, double-, and N-variant constructs. It runs a splice-site predictor
(SpliceAI, OpenSpliceAI, Pangolin, or Spliceformer) under each variant context,
computes per-site residuals against the additive expectation, and applies a
crisp 4-class mechanistic classifier.

Implements the algorithms from:

1. *Detecting and understanding meaningful cancerous mutations based on computational models of mRNA splicing* — Lynn & Tuller, *npj Systems Biology* 2024.
2. *Large-scale insight into missplicing, intra-gene epistasis and its relevance to human cancer* — in preparation.

```bash
pip install oncosplice[spliceai_pytorch]
```

## What it does

Given two (or more) genomic variants in the same gene, oncosplice answers:

- **Single-variant impact.** For each mutation alone, how much does it perturb every splice site in the gene? `analyze_single()`.
- **Joint behavior.** What does splicing look like when both mutations co-occur, and how does that compare to the additive prediction? `analyze_pair()` / `analyze_multi()`.
- **Mechanism.** Is the joint effect a *synergistic* gain (joint > additive), a *rescue* (single disrupts, joint restores WT), a *compounding* sub-additive stack, or just dominance / noise? Per-site and pair-level classification.
- **Bulk classification.** Run the same analysis over a DataFrame of hundreds of thousands of pairs with per-gene scheduling, batched inference, and resumable checkpointing. `scan()` / `classify_dataframe()`.

## Install

```bash
# Recommended — original SpliceAI weights, PyTorch backbone (no TF dependency)
pip install oncosplice[spliceai_pytorch]

# Or pick another engine
pip install oncosplice[openspliceai]   # OpenSpliceAI (MANE-trained, retrained)
pip install oncosplice[pangolin]       # Pangolin (40-model multi-tissue)
pip install oncosplice[spliceformer]   # Spliceformer (40k transformer)
pip install oncosplice[all]            # all 4 production engines

# Optional add-ons
pip install oncosplice[protein]        # protein-divergence score (Lynn & Tuller 2024)
```

Core requires `numpy`, `pandas`, `matplotlib`, `biopython`, `seqmat`. The
classification core (`analyze_pair`, `scan`, `classify_dataframe`) has **no
`geney` dependency** — `geney` is only needed for the protein-divergence score
path (`[protein]` extra).

**Model weights download automatically** from the [Hugging Face Hub](https://huggingface.co/nicolynnvila/oncosplice-weights)
on first use and are cached in `~/.oncosplice/weights/` — no manual step. Set
`ONCOSPLICE_AUTO_DOWNLOAD=0` to require an explicit `oncosplice-download-weights`
instead (useful offline / in CI).

## Highlights

- **Four production engines under one interface** — SpliceAI (PyTorch port,
  numerically identical to Keras), OpenSpliceAI, Pangolin, Spliceformer. Swap
  with one string. Cross-engine ensembling via `ensemble:a,b,c`.
- **Four-class mechanistic classifier** — rescue / cryptic rescue / deletion
  synergy / cryptic synergy, defined on probability bands with a hard
  WT-vs-annotation prerequisite that filters predictor noise.
- **TCGA-scale runner** — `classify_dataframe()` does per-gene grouping +
  batched inference + resumable checkpointing. ~23× faster than per-pair after
  the 3.2.0 vectorization; 800k pairs in ~22 hours on an L40S.
- **Numerical parity tests** between Keras SpliceAI and the PyTorch port so
  the migration is auditable.
- **Pure-python scoring core** (`oncosplice.scoring`) with no model
  dependencies — usable as a library in other splicing-prediction stacks.

## Quickstart

```python
from oncosplice import OncospliceEngine

eng = OncospliceEngine(splicing_engine="spliceai_pytorch")

# Single variant — does this mutation cause missplicing?
single = eng.analyze_single("KRAS:12:25227344:A:T")
print(single.summary())
print(single.missplicing.to_dataframe())     # missed + discovered sites
single.plot_missplicing()

# Pair — what happens when both mutations co-occur?
pair = eng.analyze_pair("KRAS:12:25227343:G:T", "KRAS:12:25227344:A:T")
print(pair.pair_classification)               # → "rescue"
print(pair.epistatic_sites())                 # only the syn/rescue/comp sites
pair.plot_case_study()                        # the bar figure

# N-variant (higher-order)
multi = eng.analyze_multi([
    "KRAS:12:25227343:G:T",
    "KRAS:12:25227344:A:T",
    "KRAS:12:25227345:G:C",
])
```

### Bulk classification of a DataFrame

```python
import pandas as pd
df = pd.read_csv("pairs.csv")   # column: epistasis_id (e.g. "GENE:CHR:POS:REF:ALT|GENE:CHR:POS:REF:ALT")

out = eng.classify_dataframe(
    df, epistasis_id_col="epistasis_id",
    checkpoint_path="results.csv",
)
# adds: pair_classification, max_abs_residual, max_abs_event_delta,
#       n_del_syn, n_cryp_syn, n_rescue, n_cryp_rescue, engine, error
```

Per-gene grouping + batched `scan()` underneath — typically 10–40× faster than the per-pair path on TCGA-shaped datasets. The runner is resume-safe (re-running with the same checkpoint path skips already-done pairs) and emits both per-pair and per-single CSVs.

### Engine-only API (no geney needed)

```python
from oncosplice.engines import get_predictor, list_available_engines
print(list_available_engines())

p = get_predictor("spliceai_pytorch")
pred = p.predict(padded_sequence)  # → SplicingPrediction(acceptor, donor)
```

## The classifier — 4 mechanism classes

At every splice site, given four predicted probabilities `ref`, `mut1`,
`mut2`, `event` (all in [0, 1]) and the annotation flag, we test four
mutually-exclusive rules. The residual `expected − event` (or `event − expected`,
depending on direction) plus the band-membership of `ref`, `mut1`, `mut2`,
`event` decide the class. `expected = mut1 + mut2 − ref` is the additive null.

**Thresholds (one set, used everywhere):**

| Symbol | Value | Meaning |
|---|---|---|
| `HIGH` | 0.50 | "site present" (includes alt-spliced sites) |
| `LOW` | 0.05 | "site absent" |
| `RES` | 0.10 | minimum residual magnitude |
| `NEAR_WT` | 0.20 | `|event − ref|` tolerance for rescue |

**Hard prerequisite — WT prediction must agree with annotation.** Every rule
first checks that the engine's wild-type prediction is consistent with the
annotation: `annotated == True ⇒ ref ≥ HIGH`, `annotated == False ⇒ ref ≤ LOW`.
Sites where the engine disagrees with the annotation are dropped as
non-epistatic without consulting the mutations. This is the noise filter.

### The four rules

| Class | When the site is annotated (`ref ≥ HIGH`) | Rule | Residual |
|---|---|---|---|
| **rescue** | one single deletes, joint restores | `min(mut1, mut2) ≤ ref − HIGH` ∧ `|event − ref| ≤ NEAR_WT` ∧ `event − min(mut1, mut2) ≥ RES` | `rescue_residual = event − min(mut1, mut2)` |
| **deletion synergy** | both singles preserve, joint destroys | `min(mut1, mut2) ≥ HIGH` ∧ `ref − event ≥ RES` ∧ `expected − event ≥ RES` | `synergy_residual = expected − event` |

| Class | When the site is not annotated (`ref ≤ LOW`) | Rule | Residual |
|---|---|---|---|
| **cryptic rescue** | one single creates, joint silences | `max(mut1, mut2) ≥ HIGH` ∧ `event ≤ LOW` ∧ `max(mut1, mut2) − event ≥ RES` | `rescue_residual = max(mut1, mut2) − event` |
| **cryptic synergy** | both silent, joint creates | `max(mut1, mut2) ≤ LOW` ∧ `event ≥ HIGH` ∧ `event − expected ≥ RES` | `synergy_residual = event − expected` |

Anything else → **non-epistatic**.

### Numeric example

```
# annotated acceptor in INPP5J — spliceai_pytorch
ref = 0.972   annotated = True
m1  = 0.658   (m1 alone preserves: 0.658 ≥ 0.50)
m2  = 0.841   (m2 alone preserves: 0.841 ≥ 0.50)
event   = 0.339
expected = m1 + m2 - ref = 0.527

# ref ≥ HIGH ✓ and annotated ✓                   → annotated branch
# min(m1, m2) = 0.658 ≥ HIGH ✓                    → not rescue (singles preserve)
# ref - event = 0.633 ≥ RES (0.10) ✓
# expected - event = 0.188 ≥ RES (0.10) ✓         → deletion_synergy
# synergy_residual = 0.188
```

### Pair-level aggregation

A pair's overall label is the class of the splice site with the *largest*
mechanism residual (rescue or synergy). Ties break by class priority:
`deletion_synergy > cryptic_synergy > rescue > cryptic_rescue > non-epistatic`.
The full per-site breakdown is always retained in `pair.site_residuals`.

## Available splicing engines

| Name | Architecture | Notes |
|---|---|---|
| `spliceai_pytorch` (default for production) | Original SpliceAI weights (Jaganathan 2019), plain-ReLU PyTorch architecture | Numerically identical to Keras SpliceAI, ~2.5× faster, no TF dependency |
| `openspliceai` | OpenSpliceAI PyTorch port, MANE-trained 5-model ensemble | Independent retrain; differs from Keras SpliceAI in fine numerics |
| `pangolin` | 40-model multi-tissue PyTorch ensemble (Zeng & Li 2022) | Tissue-specific splice usage |
| `spliceformer` | 40k-context transformer ensemble (Jónsson 2024) | Long-range context; requires the Spliceformer repo |
| `spliceai_keras` | Original Illumina `.h5` weights | **Reference only** — prefer `spliceai_pytorch` |
| `ensemble:a,b,c` / `average` | Mean probabilities across N constituent engines | Cross-engine consensus |

## Package layout

```
oncosplice/
├── engine.py             # OncospliceEngine — orchestrator (analyze_single/pair/multi, scan, classify_dataframe)
├── results.py            # typed dataclasses: SingleVariantResult, DoubleVariantResult, MultiVariantResult
├── variants.py           # Variant + VariantPair (no geney dependency)
├── viz.py                # plot_case_study + supporting bar figures
├── engines/              # standalone splice-site predictor adapters (uniform interface)
│   ├── base.py
│   ├── spliceai_pytorch.py
│   ├── openspliceai.py
│   ├── pangolin.py
│   ├── spliceformer.py
│   ├── spliceai_keras.py
│   └── ensemble.py
├── scoring/              # pure-Python scoring primitives
│   ├── splicing.py
│   ├── epistasis.py      # the 3-bucket classifier + vectorized residual computation
│   ├── oncosplice.py     # protein-divergence Oncosplice score
│   └── fingerprint.py    # splicing-outcome hashing
└── weights/              # weight-resolution + downloader CLI
```

## Examples

See `examples/`:

- `KRAS_rescue.ipynb` — a canonical KRAS donor disrupted by mut1 alone, restored by the joint event. The mechanism the classifier surfaces as **rescue**.
- `CREBBP_synergistic.ipynb` — the joint event activates one cryptic acceptor (synergy) while rescuing another from each single's activation (rescue). The classifier reports the dominant **synergistic** call with the rescue site preserved in the per-site table.

## Testing

```bash
pytest tests/                              # full suite
pytest tests/test_scoring.py               # classifier + residual rules
pytest tests/test_spliceai_equivalence.py  # Keras ↔ PyTorch numerical parity
```

## Citing

If you use this code in a published analysis, please cite the two papers above.

## License

MIT — see [LICENSE](LICENSE).
