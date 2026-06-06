"""Spliceformer (Jónsson et al., Comms Bio 2024) adapter.

Spliceformer is a transformer-based splice-site predictor with a 40 kb input
context (CL_max=40000 → 20,000 bp on each side of the prediction region).

Self-contained: the model definition is vendored (MIT) at
:mod:`oncosplice.engines._vendor.spliceformer`, and the released checkpoint
ensemble is fetched into the oncosplice weight cache. No GitHub clone or
``SPLICEFORMER_REPO_DIR`` is required.

This adapter:

1. Resolves the weight directory via the oncosplice weight resolver
   (``~/.oncosplice/weights/spliceformer/`` etc.).
2. Loads the transformer-encoder checkpoint ensemble (files matching
   ``transformer_encoder_45k_*``) using the vendored ``SpliceFormer`` class.
3. Returns ``(acceptor, donor)`` probability arrays sliced to the biological
   region, matching the contract of the other adapters.
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence

import numpy as np

from .base import SplicingPrediction, SplicingPredictor

# Pin to the base 45k ensemble only. The upstream repo also ships task-finetuned
# checkpoints (e.g. transformer_encoder_45k_finetune_rnasplice-blood_*); a looser
# glob would silently average those into the general-purpose ensemble.
_DEFAULT_GLOB = "transformer_encoder_45k_171022_*"

_IN_MAP = np.asarray(
    [[0, 0, 0, 0],
     [1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 1]],
    dtype=np.float32,
)


class Spliceformer(SplicingPredictor):
    """Spliceformer 40k-context transformer ensemble."""

    name = "spliceformer"
    _CONTEXT = 20000  # CL_max=40000 → 20,000 each side

    _models = None
    _device = None

    def __init__(
        self,
        weights_dir: Optional[str | os.PathLike] = None,
        device: Optional[str] = None,
        model_glob: str = _DEFAULT_GLOB,
    ):
        from pathlib import Path
        self._dir_override = Path(weights_dir) if weights_dir else None
        self._device_override = device
        self._model_glob = model_glob

    @property
    def context_length(self) -> int:
        return self._CONTEXT

    def is_available(self) -> bool:
        try:
            import einops  # noqa: F401  (required by SpliceFormer)
            import torch  # noqa: F401
        except ImportError:
            return False
        return self._resolve_weights_dir() is not None

    # ------------------------------------------------------------------
    def _resolve_weights_dir(self):
        from pathlib import Path
        if self._dir_override is not None and self._dir_override.exists():
            return self._dir_override
        from ..weights import resolve_dir as _resolve
        d = _resolve("spliceformer")
        if d is not None and any(Path(d).glob(self._model_glob)):
            return Path(d)
        return None

    def _pick_device(self):
        import sys

        import torch
        if self._device_override:
            return torch.device(self._device_override)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if sys.platform == "darwin" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            try:
                torch.tensor([1.0], device="mps")
                return torch.device("mps")
            except Exception:
                pass
        return torch.device("cpu")

    def _ensure_loaded(self):
        if Spliceformer._models is not None:
            return Spliceformer._models, Spliceformer._device

        import torch

        from ._vendor.spliceformer import SpliceFormer

        weights_dir = self._resolve_weights_dir()
        if weights_dir is None and self._dir_override is None:
            from ..weights import ensure_dir
            weights_dir = ensure_dir("spliceformer")  # auto-downloads on a miss
        if weights_dir is None:
            raise RuntimeError(
                "Spliceformer weights not found. Run "
                "`oncosplice-download-weights spliceformer` or set "
                "ONCOSPLICE_WEIGHTS_DIR. Expected checkpoint files matching "
                f"{self._model_glob!r} in the spliceformer weight directory."
            )

        device = self._pick_device()
        weight_paths = sorted(weights_dir.glob(self._model_glob))

        models = []
        for wp in weight_paths:
            m = SpliceFormer(CL_max=2 * self._CONTEXT, determenistic=True).to(device)
            state = torch.load(wp, map_location=device, weights_only=False)
            # Strip "module." prefix if checkpoint came from DataParallel
            cleaned = {k.removeprefix("module."): v for k, v in state.items()}
            try:
                m.load_state_dict(cleaned, strict=True)
            except RuntimeError:
                # Some checkpoints carry an extra policy/value head the strict
                # load rejects. Fall back to non-strict, but surface exactly what
                # didn't match — a mismatch in the *prediction* head (conv_final
                # / SpliceAI backbone) means garbage output, not a benign skip.
                incompat = m.load_state_dict(cleaned, strict=False)
                serious = [
                    k for k in (list(incompat.missing_keys) + list(incompat.unexpected_keys))
                    if not k.startswith("policy")
                ]
                if serious:
                    import warnings
                    warnings.warn(
                        f"Spliceformer checkpoint {wp.name}: non-strict load with "
                        f"prediction-head key mismatches {serious[:8]}"
                        f"{'...' if len(serious) > 8 else ''} — output may be unreliable.",
                        RuntimeWarning, stacklevel=2,
                    )
            m.eval()
            models.append(m)
        if not models:
            raise RuntimeError(f"No Spliceformer weights matched {self._model_glob} in {weights_dir}")
        Spliceformer._models = models
        Spliceformer._device = device
        return models, device

    # ------------------------------------------------------------------
    @staticmethod
    def _one_hot(seq: str) -> np.ndarray:
        """Encode ACGTN into (L, 4)."""
        allowed = set("ACGT")
        seq = "".join(c if c in allowed else "N" for c in seq.upper())
        seq = seq.translate(str.maketrans({"A": "1", "C": "2", "G": "3", "T": "4", "N": "0"}))
        arr = np.fromiter(map(int, seq), dtype=np.int8)
        return _IN_MAP[arr]  # (L, 4)

    def predict_one(self, sequence: str) -> SplicingPrediction:
        return self.predict_batch([sequence])[0]

    def predict_batch(self, sequences: Sequence[str]) -> List[SplicingPrediction]:
        if not sequences:
            return []
        import torch
        models, device = self._ensure_loaded()
        cl = self._CONTEXT

        results: List[SplicingPrediction] = []
        for seq in sequences:
            # (L, 4) → (4, L) → (1, 4, L)
            x = self._one_hot(seq).T.astype(np.float32)
            x = torch.from_numpy(x[None, :, :]).to(device)

            accum = None
            with torch.no_grad():
                for m in models:
                    # SpliceFormer.forward returns
                    # (out, acceptor_actions, donor_actions, acc_log_probs, don_log_probs)
                    out, *_ = m(x)        # out: (1, 3, L - CL_max)
                    out = out.detach().cpu().numpy()
                    accum = out if accum is None else accum + out
            y = accum / len(models)        # (1, 3, L_out)

            # The model with crop=True already drops the CL_max context on each
            # side, so y has length len(seq) - 2*cl — matching our contract.
            y0 = y[0]                       # (3, L_out)
            if y0.shape[1] != len(seq) - 2 * cl:
                raise RuntimeError(
                    f"unexpected Spliceformer output length {y0.shape[1]} for "
                    f"input len(seq)={len(seq)} (expected {len(seq) - 2*cl})"
                )
            results.append(SplicingPrediction(acceptor=y0[1, :], donor=y0[2, :]))
        return results
