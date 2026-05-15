"""SpliceAI original (Jaganathan et al. 2019, Keras/TensorFlow) adapter.

Wraps the upstream ``spliceai`` PyPI package, which loads the 5-model Keras
ensemble bundled with the package (``spliceai/models/spliceai*.h5``). Weights
are cached at class level after first load.

Use this when you specifically need the original reference predictions —
:class:`OpenSpliceAI` is faster on modern hardware (PyTorch + MPS/CUDA) and
typically produces near-identical outputs.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .base import SplicingPredictor, SplicingPrediction


class SpliceAIKeras(SplicingPredictor):
    """Original Illumina SpliceAI (TF/Keras) ensemble."""

    name = "spliceai_keras"
    _CONTEXT = 5000  # 5,000 bp on each side
    _models = None   # class-level cache of the 5 Keras models

    @property
    def context_length(self) -> int:
        return self._CONTEXT

    def is_available(self) -> bool:
        try:
            import spliceai  # noqa: F401
            from keras.models import load_model  # noqa: F401
            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    def _ensure_loaded(self):
        if SpliceAIKeras._models is not None:
            return SpliceAIKeras._models
        from pathlib import Path
        from keras.models import load_model
        from ..weights import resolve_dir as _resolve

        # 1) centralised resolver (~/.oncosplice/weights/spliceai_keras/)
        d = _resolve("spliceai_keras")
        if d is not None:
            paths = [d / f"spliceai{i}.h5" for i in range(1, 6)]
            if all(p.exists() for p in paths):
                SpliceAIKeras._models = [load_model(str(p)) for p in paths]
                return SpliceAIKeras._models
        # 2) upstream pip package (spliceai)
        try:
            from pkg_resources import resource_filename
            paths = [resource_filename("spliceai", f"models/spliceai{i}.h5") for i in range(1, 6)]
            SpliceAIKeras._models = [load_model(p) for p in paths]
            return SpliceAIKeras._models
        except Exception as e:
            raise RuntimeError(
                "SpliceAI Keras weights not found. Either `pip install spliceai` "
                "or run `oncosplice-download-weights spliceai_keras`."
            ) from e

    @staticmethod
    def _one_hot(sequence: str) -> np.ndarray:
        """One-hot encode an ACGTN sequence into shape (1, L, 4)."""
        # SpliceAI's encoding: A=0, C=1, G=2, T=3, N=all-zeros
        table = {"A": 0, "C": 1, "G": 2, "T": 3}
        L = len(sequence)
        x = np.zeros((L, 4), dtype=np.float32)
        for i, b in enumerate(sequence):
            idx = table.get(b.upper(), None)
            if idx is not None:
                x[i, idx] = 1.0
        return x[np.newaxis, :, :]  # (1, L, 4)

    # ------------------------------------------------------------------
    def predict_one(self, sequence: str) -> SplicingPrediction:
        models = self._ensure_loaded()
        x = self._one_hot(sequence)  # (1, L, 4)
        # Average across the 5-model ensemble; each model returns (1, L', 3)
        # where L' = L - 2*context_length and the 3 channels are
        # (no-site, acceptor, donor).
        y = np.mean([m.predict(x, verbose=0) for m in models], axis=0)[0]
        return SplicingPrediction(acceptor=y[:, 1], donor=y[:, 2])

    def predict_batch(self, sequences: Sequence[str]) -> List[SplicingPrediction]:
        # Keras predict on the concatenated batch — slightly faster than the
        # per-sequence loop for short calls; Keras handles internal batching.
        if not sequences:
            return []
        models = self._ensure_loaded()
        X = np.concatenate([self._one_hot(s) for s in sequences], axis=0)
        y = np.mean([m.predict(X, verbose=0) for m in models], axis=0)
        return [SplicingPrediction(acceptor=y[i, :, 1], donor=y[i, :, 2])
                for i in range(y.shape[0])]
