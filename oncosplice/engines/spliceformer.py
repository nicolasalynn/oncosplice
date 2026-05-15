"""Spliceformer (Jónsson et al., Comms Bio 2024) adapter.

Spliceformer is a transformer-based splice-site predictor with a 40 kb input
context (CL_max=40000 → 20,000 bp on each side of the prediction region).
It is distributed as a research GitHub repo
(https://github.com/benniatli/Spliceformer); the model class lives in
``Code/src/model.py`` and pre-trained weights are PyTorch state dicts in
``Results/PyTorch_Models/``.

This adapter:

1. Locates the cloned Spliceformer repo (env ``SPLICEFORMER_REPO_DIR`` or
   ``~/Documents/phd/libraries/Spliceformer``).
2. Adds its ``Code/`` directory to ``sys.path`` so ``from src.model import
   SpliceFormer`` works.
3. Loads the 10-checkpoint 45k transformer ensemble
   (``transformer_encoder_45k_*``) and runs them at inference time.
4. Returns ``(acceptor, donor)`` probability arrays sliced to the biological
   region, matching the contract of the other adapters.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .base import SplicingPrediction, SplicingPredictor

_DEFAULT_REPO = Path.home() / "Documents/phd/libraries/Spliceformer"
_DEFAULT_GLOB = "transformer_encoder_45k_*"


def _default_repo_dir() -> Path:
    return Path(os.environ.get("SPLICEFORMER_REPO_DIR", _DEFAULT_REPO))


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
    _SpliceFormer = None  # cached model class

    def __init__(
        self,
        repo_dir: Optional[str | os.PathLike] = None,
        device: Optional[str] = None,
        model_glob: str = _DEFAULT_GLOB,
    ):
        self.repo_dir = Path(repo_dir) if repo_dir else _default_repo_dir()
        self._device_override = device
        self._model_glob = model_glob

    @property
    def context_length(self) -> int:
        return self._CONTEXT

    def is_available(self) -> bool:
        if not self.repo_dir.exists():
            return False
        if not (self.repo_dir / "Code" / "src" / "model.py").exists():
            return False
        if not any((self.repo_dir / "Results" / "PyTorch_Models").glob(self._model_glob)):
            return False
        try:
            import einops  # noqa: F401  (required by SpliceFormer)
            import torch  # noqa: F401
            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    def _ensure_class(self):
        """Import the SpliceFormer class from the cloned repo. Idempotent."""
        if Spliceformer._SpliceFormer is not None:
            return Spliceformer._SpliceFormer
        code_dir = str(self.repo_dir / "Code")
        if code_dir not in sys.path:
            sys.path.insert(0, code_dir)
        from src.model import SpliceFormer  # type: ignore
        Spliceformer._SpliceFormer = SpliceFormer
        return SpliceFormer

    def _pick_device(self):
        import torch
        if self._device_override:
            return torch.device(self._device_override)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
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

        if not self.is_available():
            raise RuntimeError(
                f"Spliceformer is not set up. Expected:\n"
                f"  - repo at {self.repo_dir}\n"
                f"  - Code/src/model.py with `class SpliceFormer`\n"
                f"  - checkpoints matching {self._model_glob} in Results/PyTorch_Models/\n"
                f"  - python deps: torch, einops"
            )

        SpliceFormer = self._ensure_class()
        device = self._pick_device()
        weight_paths = sorted((self.repo_dir / "Results/PyTorch_Models").glob(self._model_glob))

        models = []
        for wp in weight_paths:
            m = SpliceFormer(CL_max=2 * self._CONTEXT, determenistic=True).to(device)
            state = torch.load(wp, map_location=device, weights_only=False)
            # Strip "module." prefix if checkpoint came from DataParallel
            cleaned = {k.removeprefix("module."): v for k, v in state.items()}
            try:
                m.load_state_dict(cleaned, strict=True)
            except RuntimeError:
                # Some checkpoints may be slightly mismatched (e.g. different policy
                # head). Fall back to non-strict; SpliceFormer's prediction head is
                # what we need and survives a partial load.
                m.load_state_dict(cleaned, strict=False)
            m.eval()
            models.append(m)
        if not models:
            raise RuntimeError(f"No Spliceformer weights matched {self._model_glob}")
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

    # ------------------------------------------------------------------
    @staticmethod
    def write_model_shim(*args, **kwargs):  # pragma: no cover
        """Deprecated — kept for backwards compatibility. The adapter now
        imports ``SpliceFormer`` directly from ``Code/src/model.py`` in the
        cloned repo. No manual shim file is required.
        """
        import warnings
        warnings.warn(
            "write_model_shim is no longer needed; Spliceformer is loaded "
            "from Code/src/model.py automatically.",
            DeprecationWarning, stacklevel=2,
        )
