"""Verify that the original SpliceAI Keras model and SpliceAI weights *translated*
to PyTorch produce numerically equivalent predictions on the same sequence.

What this test does NOT compare:
  - Keras SpliceAI vs. OpenSpliceAI's own MANE-trained weights (those are
    different training runs and *will* differ — and loading the translated
    weights into OpenSpliceAI's LeakyReLU architecture is also not equivalent).

What this test DOES compare:
  - The five canonical SpliceAI .h5 Keras weights, run in TensorFlow.
  - The same five SpliceAI weights translated to PyTorch state dicts and run in
    oncosplice's plain-ReLU SpliceAI architecture (the ``SpliceAIPyTorch`` engine).

Because the architecture, layer order, and weight values match across the two
framework runs, any non-zero difference is purely numerical (float32 BatchNorm
epsilon, conv op ordering, etc.). We expect:

- Max absolute difference on acceptor/donor probabilities < ~5e-3
- Mean absolute difference < ~1e-3

To run this test you need both:
  1. SpliceAI installed (`pip install spliceai tensorflow`) with its .h5 weights.
  2. The translated PyTorch weights in the oncosplice cache
     (`oncosplice-download-weights spliceai_pytorch`).

The test skips cleanly if either backend is missing.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pytest


SEED = 0
SEQ_LEN_BIOLOGICAL = 1000              # length of biological region to test
CONTEXT = 5000                          # SpliceAI / OpenSpliceAI half-context

# Tolerances — match the documented OpenSpliceAI port accuracy.
MAX_ABS_TOL  = 5e-3
MEAN_ABS_TOL = 1e-3


def _random_acgt(length: int, *, seed: int = SEED) -> str:
    """Deterministic random ACGT string for reproducibility."""
    rng = np.random.default_rng(seed)
    return "".join(rng.choice(list("ACGT"), size=length))


def _padded_input() -> str:
    """The biological region plus `CONTEXT` N-padding on each side."""
    biological = _random_acgt(SEQ_LEN_BIOLOGICAL)
    return "N" * CONTEXT + biological + "N" * CONTEXT


@pytest.fixture(scope="module")
def keras_predictor():
    from oncosplice.engines import SpliceAIKeras
    p = SpliceAIKeras()
    if not p.is_available():
        pytest.skip("SpliceAI (Keras/TF) backend not installed.")
    return p


@pytest.fixture(scope="module")
def torch_predictor():
    """The **Keras-translated** SpliceAI weights run in oncosplice's plain-ReLU
    SpliceAI PyTorch architecture (the ``SpliceAIPyTorch`` engine). This is the
    only comparison that is numerically equivalent to Keras SpliceAI — loading
    these weights into OpenSpliceAI's *different* (LeakyReLU, MANE-trained)
    architecture would not be.
    """
    from oncosplice.engines import SpliceAIPyTorch

    p = SpliceAIPyTorch()
    if not p.is_available():
        pytest.skip(
            "SpliceAI→PyTorch translated weights not present. "
            "Run `oncosplice-download-weights spliceai_pytorch` or "
            "place them in ~/.oncosplice/weights/spliceai_pytorch/."
        )
    return p


@pytest.fixture(scope="module")
def shared_sequence():
    return _padded_input()


def _safely_predict(predictor, sequence):
    try:
        return predictor.predict(sequence)
    except RuntimeError as e:
        pytest.skip(f"{predictor.name} could not load weights: {e}")


def test_shapes_match(keras_predictor, torch_predictor, shared_sequence):
    k = _safely_predict(keras_predictor, shared_sequence)
    t = _safely_predict(torch_predictor, shared_sequence)
    assert k.acceptor.shape == t.acceptor.shape, \
        f"acceptor shapes differ: keras={k.acceptor.shape} torch={t.acceptor.shape}"
    assert k.donor.shape == t.donor.shape, \
        f"donor shapes differ: keras={k.donor.shape} torch={t.donor.shape}"
    assert k.acceptor.shape == (SEQ_LEN_BIOLOGICAL,), \
        f"expected length {SEQ_LEN_BIOLOGICAL}, got {k.acceptor.shape}"


def test_acceptor_probabilities_match(keras_predictor, torch_predictor, shared_sequence):
    k = _safely_predict(keras_predictor, shared_sequence)
    t = _safely_predict(torch_predictor, shared_sequence)
    diff = np.abs(k.acceptor - t.acceptor)
    print(f"\n  acceptor  max|Δ|={diff.max():.6f}  mean|Δ|={diff.mean():.6f}")
    assert diff.max()  < MAX_ABS_TOL,  f"acceptor max|Δ| = {diff.max()} exceeds {MAX_ABS_TOL}"
    assert diff.mean() < MEAN_ABS_TOL, f"acceptor mean|Δ| = {diff.mean()} exceeds {MEAN_ABS_TOL}"


def test_donor_probabilities_match(keras_predictor, torch_predictor, shared_sequence):
    k = _safely_predict(keras_predictor, shared_sequence)
    t = _safely_predict(torch_predictor, shared_sequence)
    diff = np.abs(k.donor - t.donor)
    print(f"\n  donor     max|Δ|={diff.max():.6f}  mean|Δ|={diff.mean():.6f}")
    assert diff.max()  < MAX_ABS_TOL,  f"donor max|Δ| = {diff.max()} exceeds {MAX_ABS_TOL}"
    assert diff.mean() < MEAN_ABS_TOL, f"donor mean|Δ| = {diff.mean()} exceeds {MEAN_ABS_TOL}"


def test_top_site_ranking_matches(keras_predictor, torch_predictor, shared_sequence):
    """Beyond per-base numerical equivalence, verify the same top-K sites get
    flagged by both backends — what actually matters for downstream calls.
    """
    k = _safely_predict(keras_predictor, shared_sequence)
    t = _safely_predict(torch_predictor, shared_sequence)
    K = 20
    for label, k_arr, t_arr in (("acceptor", k.acceptor, t.acceptor),
                                 ("donor",    k.donor,    t.donor)):
        k_top = set(np.argsort(k_arr)[-K:])
        t_top = set(np.argsort(t_arr)[-K:])
        overlap = len(k_top & t_top)
        print(f"\n  {label:9s} top-{K} overlap: {overlap}/{K}")
        assert overlap >= int(0.8 * K), \
            f"{label} top-{K} overlap = {overlap}/{K} (< 80%)"


if __name__ == "__main__":
    # Convenience: run with `python tests/test_spliceai_equivalence.py`
    sys.exit(pytest.main([__file__, "-v", "-s"]))
