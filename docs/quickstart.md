# Quickstart

## Install

```bash
pip install oncosplice[spliceai_pytorch]
```

The first call downloads model weights on demand (cached under
`~/.cache/oncosplice/`). For other engines see [Engines](engines.md).

## Single variant

```python
from oncosplice import OncospliceEngine

eng = OncospliceEngine(splicing_engine="spliceai_pytorch")

single = eng.analyze_single("KRAS:12:25227344:A:T")
print(single.summary())
print(single.missplicing.to_dataframe())
single.plot_missplicing()
```

`single.missplicing` lists every splice site whose probability moved by more
than the default Δ threshold (0.20).

## Pair (two-variant epistasis)

```python
pair = eng.analyze_pair(
    "KRAS:12:25227343:G:T",
    "KRAS:12:25227344:A:T",
)
print(pair.pair_classification)   # → "rescue"
print(pair.epistatic_sites())     # per-site syn/rescue/comp table
pair.plot_case_study()            # bar figure
```

The pair-level classification picks the dominant call by priority:
**synergistic > rescue > compounding > non-epistatic** — see [Classifier](classifier.md).

## N-variant

```python
multi = eng.analyze_multi([
    "KRAS:12:25227343:G:T",
    "KRAS:12:25227344:A:T",
    "KRAS:12:25227345:G:C",
])
```

## Bulk: TCGA-shaped DataFrames

```python
import pandas as pd
df = pd.read_csv("pairs.csv")  # column: epistasis_id "GENE:CHR:POS:REF:ALT|GENE:CHR:POS:REF:ALT"

out = eng.classify_dataframe(
    df, epistasis_id_col="epistasis_id",
    checkpoint_path="results.csv",
)
```

`classify_dataframe()` does per-gene grouping + batched inference + resumable
checkpointing. See [Bulk analysis](bulk.md) for the engine-internals.

## Engine-only API (no geney needed)

```python
from oncosplice.engines import get_predictor, list_available_engines
print(list_available_engines())

p = get_predictor("spliceai_pytorch")
pred = p.predict(padded_sequence)  # → SplicingPrediction(acceptor, donor)
```

Useful when you only want splice-site probabilities and don't need the
classifier or `geney`/`seqmat` dependencies.
