# oncosplice

[![CI](https://github.com/nicolaslynn/oncosplice/actions/workflows/ci.yml/badge.svg)](https://github.com/nicolaslynn/oncosplice/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Sequence-level pipeline for splicing-epistasis analysis** of single-, double-, and N-variant constructs. Computes splice-site probabilities under each context, classifies the joint effect against the additive prediction, and identifies the mechanism (synergy, rescue, compounding).

Implements the algorithms from:

1. *Detecting and understanding meaningful cancerous mutations based on computational models of mRNA splicing* (Lynn & Tuller, npj Systems Biology 2024) — the Oncosplice scoring pipeline.
2. *Large-scale insight into missplicing, intra-gene epistasis and its relevance to human cancer* (in prep) — the splicing-epistasis layer for multi-variant constructs.

## What it does

Given two (or more) genomic variants in the same gene, oncosplice answers:

- **Single-variant impact.** For each mutation alone, how much does it perturb every splice site in the gene? `analyze_single()`.
- **Joint behavior.** What does splicing look like when both mutations co-occur, and how does that compare to the additive prediction? `analyze_pair()` / `analyze_multi()`.
- **Mechanism.** Is the joint effect a *synergistic* gain (joint > additive), a *rescue* (single disrupts, joint restores WT), a *compounding* sub-additive stack, or just dominance / noise? Per-site and pair-level classification.
- **Bulk classification.** Run the same analysis over a DataFrame of hundreds of thousands of pairs with per-gene scheduling, batched inference, and resumable checkpointing. `scan()` / `classify_dataframe()`.

## Install

```bash
pip install -e .                # core (no splice engine)
pip install -e .[all]           # all 4 production engines
pip install -e .[openspliceai]  # OpenSpliceAI alone
pip install -e .[spliceai_pytorch]  # original SpliceAI weights, PyTorch architecture
pip install -e .[pangolin]      # Pangolin
pip install -e .[spliceformer]  # Spliceformer 40k transformer
```

Core requires: `geney`, `seqmat`, `numpy`, `pandas`, `matplotlib`, `biopython`. The package preserves your env if installed with `--no-deps`.

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
#       n_syn, n_rescue, n_compound, engine, error
```

Per-gene grouping + batched `scan()` underneath — typically 10–40× faster than the per-pair path on TCGA-shaped datasets. The runner is resume-safe (re-running with the same checkpoint path skips already-done pairs) and emits both per-pair and per-single CSVs.

### Engine-only API (no geney needed)

```python
from oncosplice.engines import get_predictor, list_available_engines
print(list_available_engines())

p = get_predictor("spliceai_pytorch")
pred = p.predict(padded_sequence)  # → SplicingPrediction(acceptor, donor)
```

## The classifier (3 mechanism buckets + dominance fallback)

The classifier anchors on three deltas at each splice site: `d1 = mut1 − ref`, `d2 = mut2 − ref`, `de = event − ref`. The residual is the *signed excess of the joint over the additive expectation*: `residual = de − (d1 + d2)`. We deliberately do **not** key the rules on `|residual|` alone — probabilities saturate near 0 and 1, so a large `|residual|` is often a boundary artifact rather than biology.

| Class | Rule (priority order, top first) | Mechanism |
|---|---|---|
| **rescue** | worst single ≥ 0.30 in magnitude; joint ≤ 0.20 of WT; joint at least 0.15 closer to WT than worst; joint on the same side of WT as worst (tiny overshoots permitted). | One single substantially perturbs the splice site, the joint restores it (or near-WT). |
| **synergistic — flip** | worst and joint on opposite sides; joint ≥ 0.15; joint differs from EVERY single by ≥ 0.15. | Singles push one way, joint produces an emergent opposite effect — neither single alone resembles the joint. |
| **synergistic — emergent at edge** | ref ≤ 0.10 or ≥ 0.90; singles barely move; joint Δ ≥ 0.10. | The site was pegged at "off" or "on", neither single budges it, but the joint creates a discrete (logit-scale meaningful) change. |
| **synergistic — super-additive** | joint > worst; joint > additive; (joint − additive) > 0.25 × worst. | Joint clearly exceeds both the strongest single and the additive prediction. |
| **compounding** | both \|d\| ≥ 0.20; joint > worst; joint > additive; (joint − additive) ≤ 0.25 × worst. | Both mutations contribute meaningfully, joint reflects stacked (≈ additive) effect. |
| **non-epistatic** | else. | Includes saturation artifacts, redundant disruption, dominance of one single, and noise. |

Pair-level call: descending priority **synergistic > rescue > compounding > non-epistatic** over the per-site classifications.

The classifier deliberately does **not** include an "antagonistic" bucket. Cases that look antagonistic under a strict `|residual|` rule almost always turn out to be saturation artifacts (joint pegged at 1.0 when additive predicts > 1.0) and are correctly reported as non-epistatic.

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
