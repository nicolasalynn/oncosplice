"""Pangolin (Zeng & Li 2022) splice-site predictor adapter.

Self-contained: the network is an MIT-clean re-implementation of the published
Pangolin architecture (see :mod:`oncosplice.engines._pangolin_arch`), and the
trained weights are fetched into the oncosplice weight cache. No GitHub clone
and no ``geney`` dependency are required.

Inference reproduces the established aggregation exactly:

- 40-model ensemble — weight files ``final.{j}.{i}.3`` for tissue/metric index
  ``i ∈ 0..7`` and checkpoint ``j ∈ 1..5`` — averaged together.
- The averaged output has 12 channels per position. We take, per position, the
  max across the four tissue groups:
    * acceptor ← channels ``[1, 4, 7, 10]``
    * donor    ← channels ``[2, 5, 8, 11]``

The network crops ``CL = 10000`` bp (5,000 per side) internally, so the output
already corresponds to the biological centre of the padded input — matching the
:class:`~oncosplice.engines.base.SplicingPredictor` contract.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np

from .base import SplicingPrediction, SplicingPredictor

# Per-tissue channel selections in the averaged 12-channel output.
_ACCEPTOR_CHANNELS = [1, 4, 7, 10]
_DONOR_CHANNELS = [2, 5, 8, 11]

# Ensemble layout of the released weights: index i, checkpoint j → final.{j}.{i}.3
_MODEL_INDICES = range(8)
_MODEL_CHECKPOINTS = range(1, 6)

_IN_MAP = np.asarray(
    [[0, 0, 0, 0],   # N
     [1, 0, 0, 0],   # A
     [0, 1, 0, 0],   # C
     [0, 0, 1, 0],   # G
     [0, 0, 0, 1]],  # T
    dtype=np.float32,
)


def _one_hot(seq: str) -> np.ndarray:
    """Encode ACGTN → (L, 4); unknown bases map to N (all-zero)."""
    allowed = set("ACGT")
    seq = "".join(c if c in allowed else "N" for c in seq.upper())
    seq = seq.translate(str.maketrans({"A": "1", "C": "2", "G": "3", "T": "4", "N": "0"}))
    arr = np.fromiter(map(int, seq), dtype=np.int8)
    return _IN_MAP[arr]


class Pangolin(SplicingPredictor):
    """Pangolin (multi-tissue, 40-model ensemble) adapter."""

    name = "pangolin"
    _CONTEXT = 5000  # network crops CL=10000 → 5,000 bp each side

    # Class-level caches (the ensemble is large; load once).
    _models = None
    _device = None
    _weights_dir: Path | None = None

    def __init__(self, weights_dir: str | None = None, device: str | None = None):
        self._dir_override = Path(weights_dir) if weights_dir else None
        self._device_override = device

    @property
    def context_length(self) -> int:
        return self._CONTEXT

    def is_available(self) -> bool:
        try:
            import torch  # noqa: F401
        except ImportError:
            return False
        return self._resolve_weights_dir() is not None

    # ------------------------------------------------------------------
    def _resolve_weights_dir(self) -> Path | None:
        if self._dir_override is not None and self._dir_override.exists():
            return self._dir_override
        from ..weights import resolve_dir as _resolve
        d = _resolve("pangolin")
        if d is not None:
            return Path(d)
        # Backward-compat: an upstream `pangolin` package installed locally.
        try:
            from pkg_resources import resource_filename
            cand = Path(resource_filename("pangolin", "models"))
            if cand.exists() and any(cand.glob("final.*.3")):
                return cand
        except Exception:
            pass
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
        if Pangolin._models is not None:
            return Pangolin._models, Pangolin._device

        import torch

        from ._pangolin_arch import build_pangolin_class

        weights_dir = self._resolve_weights_dir()
        if weights_dir is None and self._dir_override is None:
            from ..weights import ensure_dir
            weights_dir = ensure_dir("pangolin")  # auto-downloads on a miss
        if weights_dir is None:
            raise RuntimeError(
                "Pangolin weights not found. Run "
                "`oncosplice-download-weights pangolin` or set "
                "ONCOSPLICE_WEIGHTS_DIR. Looked in the oncosplice weight cache "
                "and any locally-installed `pangolin` package."
            )

        Pangolin_cls = build_pangolin_class()
        device = self._pick_device()

        models = []
        missing = []
        for i in _MODEL_INDICES:
            for j in _MODEL_CHECKPOINTS:
                wp = Path(weights_dir) / f"final.{j}.{i}.3"
                if not wp.exists():
                    missing.append(wp.name)
                    continue
                m = Pangolin_cls().to(device)
                m.load_state_dict(torch.load(wp, map_location=device, weights_only=True))
                m.eval()
                models.append(m)
        if not models:
            raise RuntimeError(
                f"No Pangolin weight files (final.j.i.3) found in {weights_dir}."
            )
        if missing:
            # Partial ensembles change the numbers; surface it rather than hide it.
            import warnings
            warnings.warn(
                f"Pangolin: loaded {len(models)} of 40 ensemble members; "
                f"missing {len(missing)} weight file(s): {missing[:5]}"
                f"{'...' if len(missing) > 5 else ''}",
                RuntimeWarning, stacklevel=2,
            )
        Pangolin._models = models
        Pangolin._device = device
        Pangolin._weights_dir = Path(weights_dir)
        return models, device

    # ------------------------------------------------------------------
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
            x = torch.from_numpy(_one_hot(seq).T[None, :, :]).to(device)  # (1, 4, L)
            with torch.no_grad():
                acc = None
                for m in models:
                    out = m(x).detach().cpu().numpy()  # (1, 12, L-2cl)
                    acc = out if acc is None else acc + out
            y = (acc / len(models))[0]                  # (12, L_out)

            acceptor = np.max(y[_ACCEPTOR_CHANNELS, :], axis=0).astype(np.float32)
            donor = np.max(y[_DONOR_CHANNELS, :], axis=0).astype(np.float32)

            if acceptor.shape[0] == len(seq):
                acceptor = acceptor[cl:-cl]
                donor = donor[cl:-cl]
            elif acceptor.shape[0] != len(seq) - 2 * cl:
                raise RuntimeError(
                    f"Pangolin output length {acceptor.shape[0]} matches neither "
                    f"len(seq)={len(seq)} nor len(seq)-2*cl={len(seq) - 2 * cl}"
                )
            results.append(SplicingPrediction(acceptor=acceptor, donor=donor))
        return results
