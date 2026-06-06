# Splicing engines

All four production engines speak a uniform `SplicingPredictor` interface
(input: padded one-hot sequence; output: `SplicingPrediction(acceptor, donor)`
NumPy arrays). Pick one at install time and one at runtime.

| Name | Architecture | Notes |
|---|---|---|
| `spliceai_pytorch` *(default for production)* | Original SpliceAI weights (Jaganathan et al. 2019), plain-ReLU PyTorch backbone | Numerically identical to Keras SpliceAI (verified per-base, see [parity tests](https://github.com/nicolasalynn/oncosplice/blob/main/tests/test_spliceai_equivalence.py)). ~2.5× faster, no TF dependency. |
| `openspliceai` | OpenSpliceAI PyTorch port, MANE-trained 5-model ensemble | Independent retrain; fine numerics differ from Keras SpliceAI. |
| `pangolin` | 40-model multi-tissue PyTorch ensemble (Zeng & Li 2022) | Tissue-specific splice usage. Install Pangolin from GitHub separately. |
| `spliceformer` | 40k-context transformer ensemble (Jónsson 2024) | Long-range context. Requires the Spliceformer repo. |
| `spliceai_keras` | Original Illumina `.h5` weights | **Reference only** — prefer `spliceai_pytorch`. |
| `ensemble:a,b,c` / `average` | Mean probabilities across constituent engines | Cross-engine consensus. |

## Selecting at runtime

```python
from oncosplice import OncospliceEngine

eng = OncospliceEngine(splicing_engine="spliceai_pytorch")     # default
eng = OncospliceEngine(splicing_engine="openspliceai")
eng = OncospliceEngine(splicing_engine="ensemble:spliceai_pytorch,openspliceai")
```

## Engine-only path

If you don't need the classifier or `geney`/`seqmat`:

```python
from oncosplice.engines import get_predictor, list_available_engines

print(list_available_engines())
p = get_predictor("spliceai_pytorch")
pred = p.predict(padded_sequence)
```

`pred.acceptor` and `pred.donor` are NumPy arrays of length `len(sequence) − 2·context`.

## Weight downloads

```bash
oncosplice-download-weights --engine spliceai_pytorch
oncosplice-download-weights --engine all
```

Weights are resolved lazily on first `predict()` call; the CLI is for
pre-caching in air-gapped or CI scenarios. The cache lives at
`~/.cache/oncosplice/` by default (override with `ONCOSPLICE_CACHE_DIR`).
