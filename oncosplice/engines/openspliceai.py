"""OpenSpliceAI (PyTorch port of SpliceAI) adapter.

Delegates to the upstream ``openspliceai.predict.predict()`` function, which
correctly handles:
- one-hot encoding,
- tiled windowing (CL_max=10000, SL=5000),
- the 5-model ensemble averaging,
- softmax on the logits to get probabilities.

Reinventing this path is a recipe for subtle indexing / missing-softmax bugs —
this adapter therefore *uses* it rather than rebuilding it.
"""
from __future__ import annotations

import io
import sys
from typing import List, Sequence

import numpy as np

from .base import SplicingPrediction, SplicingPredictor


class OpenSpliceAI(SplicingPredictor):
    """OpenSpliceAI PyTorch ensemble predictor (5-model ensemble)."""

    name = "openspliceai"
    _CONTEXT = 5000  # CL_max=10000 → 5,000 bp on each side of the prediction region

    # Class-level cache of the resolved model_dir (the model bundle itself is
    # cached internally by openspliceai's predict() across calls).
    _model_dir: str | None = None

    def __init__(self, device: str | None = None, model_dir: str | None = None):
        self._device_override = device
        self._model_dir_override = model_dir

    @property
    def context_length(self) -> int:
        return self._CONTEXT

    def is_available(self) -> bool:
        try:
            import openspliceai  # noqa: F401
            import torch  # noqa: F401
            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    def _ensure_model_dir(self) -> str:
        if OpenSpliceAI._model_dir is not None:
            return OpenSpliceAI._model_dir
        if self._model_dir_override is not None:
            OpenSpliceAI._model_dir = str(self._model_dir_override)
            return OpenSpliceAI._model_dir
        # Centralised resolver: oncosplice/weights → user cache → bundled
        from ..weights import resolve_dir as _resolve
        d = _resolve("openspliceai")
        if d is not None:
            OpenSpliceAI._model_dir = str(d)
            return OpenSpliceAI._model_dir
        # Fallback: ask geney for the same resolver
        try:
            from geney.splicing.engines import _get_openspliceai_model_dir as _gd
        except ImportError:
            try:
                from geney.engines import _get_openspliceai_model_dir as _gd
            except ImportError:
                raise RuntimeError(
                    "OpenSpliceAI weights not found. Run "
                    "`oncosplice-download-weights openspliceai` or set "
                    "ONCOSPLICE_WEIGHTS_DIR / OPENSPLICEAI_MODEL_DIR."
                )
        OpenSpliceAI._model_dir = _gd()
        return OpenSpliceAI._model_dir

    # ------------------------------------------------------------------
    def predict_one(self, sequence: str) -> SplicingPrediction:
        """Run openspliceai.predict on one padded sequence.

        Returns acceptor & donor probability arrays of length
        ``len(sequence) - 2 * context_length`` — the biological centre.
        """
        from openspliceai.predict.predict import predict
        model_dir = self._ensure_model_dir()

        # Suppress openspliceai's progress prints (they're verbose for short jobs).
        _stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            y = predict(sequence, model_dir, flanking_size=10000)  # (L_out, 3)
        finally:
            sys.stdout = _stdout

        # y is a torch.Tensor on the prediction device.
        y_np = y.detach().cpu().numpy() if hasattr(y, "detach") else np.asarray(y)
        if y_np.ndim != 2 or y_np.shape[1] != 3:
            raise RuntimeError(
                f"unexpected openspliceai output shape: {y_np.shape}"
            )
        # openspliceai.predict returns probabilities for the FULL input length;
        # the predictor contract is "output length = len(seq) - 2*context".
        # Slice off the context portion on each side.
        cl = self._CONTEXT
        if y_np.shape[0] == len(sequence):
            y_np = y_np[cl:-cl, :]
        elif y_np.shape[0] != len(sequence) - 2 * cl:
            raise RuntimeError(
                f"openspliceai output length {y_np.shape[0]} matches neither "
                f"len(seq)={len(sequence)} nor len(seq)-2*cl={len(sequence)-2*cl}"
            )
        return SplicingPrediction(acceptor=y_np[:, 1], donor=y_np[:, 2])

    def predict_batch(self, sequences: Sequence[str]) -> List[SplicingPrediction]:
        # openspliceai.predict() runs one sequence at a time; in practice each
        # call already batches the internal windows on GPU. Looping is fine.
        return [self.predict_one(s) for s in sequences]
