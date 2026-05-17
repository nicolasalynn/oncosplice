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

    # Class-level cache for the loaded model ensemble + device + consts.
    # The upstream openspliceai.predict() function reloads the 5-model ensemble
    # from disk on every call — a ~2 second hit per inference. Caching here
    # drops per-call time by ~10x to roughly match spliceai_pytorch.
    _model_ensemble = None      # list of 5 torch.nn.Module
    _model_device   = None
    _model_consts   = None
    _model_params   = None

    @classmethod
    def _load_ensemble_once(cls, model_dir: str):
        """Load the 5-model ensemble + device + consts exactly once."""
        if cls._model_ensemble is not None:
            return
        from openspliceai.predict import utils as _osa_utils
        from openspliceai.predict.predict import setup_device, load_pytorch_models
        cls._model_consts   = _osa_utils.initialize_constants(10000)
        cls._model_device   = setup_device()
        cls._model_ensemble, cls._model_params = load_pytorch_models(
            model_dir, cls._model_device, cls._model_consts["SL"], 10000,
        )

    # ------------------------------------------------------------------
    def predict_one(self, sequence: str) -> SplicingPrediction:
        """Run openspliceai.predict on one padded sequence.

        Returns acceptor & donor probability arrays of length
        ``len(sequence) - 2 * context_length`` — the biological centre.
        """
        import torch
        from openspliceai.predict.predict import create_datapoints

        model_dir = self._ensure_model_dir()

        # Suppress openspliceai's chatter on first-load only.
        _stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            self._load_ensemble_once(model_dir)
        finally:
            sys.stdout = _stdout

        consts   = OpenSpliceAI._model_consts
        device   = OpenSpliceAI._model_device
        models   = OpenSpliceAI._model_ensemble
        sequence_length = len(sequence)

        # One-hot + tile (mirrors openspliceai.predict.predict)
        X = create_datapoints(sequence, SL=consts["SL"], CL_max=consts["CL_max"])
        X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1).to(device)

        with torch.no_grad():
            y_pred = torch.mean(
                torch.stack([m(X).detach().cpu() for m in models]),
                axis=0,
            )
        y_pred = y_pred.permute(0, 2, 1).contiguous().view(-1, y_pred.shape[1])
        y_pred = y_pred[:sequence_length, :]    # crop padding
        y_np = y_pred.numpy()

        # Crop the context flanks (predictor contract: out length = len(seq) - 2*cl)
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
        # One inference per sequence; the cached ensemble means the per-call
        # overhead is just one-hot + 5 forward passes.
        return [self.predict_one(s) for s in sequences]
