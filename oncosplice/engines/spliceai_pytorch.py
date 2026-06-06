"""SpliceAI weights, PyTorch architecture — Keras→PyTorch translated path.

This adapter runs the original Jaganathan-2019 SpliceAI weights inside a
PyTorch model class with plain ReLU activations (matching the Keras
architecture). It is *not* the same as :class:`OpenSpliceAI`:

- :class:`OpenSpliceAI` uses the *OpenSpliceAI* PyTorch model class
  (LeakyReLU) trained on MANE. Different training run.
- :class:`SpliceAIPyTorch` uses the *original SpliceAI* weights loaded into
  a plain-ReLU PyTorch architecture. Numerically equivalent to running the
  original Keras model (to within a few × 10⁻⁵), just on PyTorch.

Weights are ``spliceai{1..5}_torch.pt`` (the Keras→PyTorch translation),
resolved via the central weight resolver: ``$ONCOSPLICE_WEIGHTS_DIR`` override →
``~/.oncosplice/weights/spliceai_pytorch/`` cache → HuggingFace auto-download.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np

from .base import SplicingPrediction, SplicingPredictor

# ----------------------------------------------------------------------------
# Model architecture — plain ReLU SpliceAI 10k
# ----------------------------------------------------------------------------

def _arch_10000():
    """Return (L, W, AR) for the 10000-nt SpliceAI architecture."""
    L = 32
    W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                    21, 21, 21, 21, 41, 41, 41, 41])
    AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                     10, 10, 10, 10, 25, 25, 25, 25])
    return L, W, AR


def _build_model_class():
    """Return the SpliceAI_ReLU model class. Imports torch lazily."""
    import torch.nn as nn
    import torch.nn.functional as F

    class _ReLUResidualUnit(nn.Module):
        def __init__(self, l, w, ar):
            super().__init__()
            self.batchnorm1 = nn.BatchNorm1d(l, eps=1e-3)
            self.batchnorm2 = nn.BatchNorm1d(l, eps=1e-3)
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()
            self.conv1 = nn.Conv1d(l, l, w, dilation=ar, padding=(w - 1) * ar // 2)
            self.conv2 = nn.Conv1d(l, l, w, dilation=ar, padding=(w - 1) * ar // 2)

        def forward(self, x, y):
            out = self.conv1(self.relu1(self.batchnorm1(x)))
            out = self.conv2(self.relu2(self.batchnorm2(out)))
            return x + out, y

    class _Cropping1D(nn.Module):
        def __init__(self, cropping):
            super().__init__()
            self.cropping = cropping

        def forward(self, x):
            c0, c1 = self.cropping
            return x[:, :, c0:-c1] if c1 > 0 else x[:, :, c0:]

    class _Skip(nn.Module):
        def __init__(self, l):
            super().__init__()
            self.conv = nn.Conv1d(l, l, 1)

        def forward(self, x, y):
            return x, self.conv(x) + y

    class SpliceAI_ReLU(nn.Module):
        """SpliceAI architecture with plain ReLU (matches Jaganathan 2019)."""

        def __init__(self, L, W, AR):
            super().__init__()
            self.initial_conv = nn.Conv1d(4, L, 1)
            self.initial_skip = _Skip(L)
            self.residual_units = nn.ModuleList()
            for i, (w, r) in enumerate(zip(W, AR)):
                self.residual_units.append(_ReLUResidualUnit(L, w, r))
                if (i + 1) % 4 == 0:
                    self.residual_units.append(_Skip(L))
            self.final_conv = nn.Conv1d(L, 3, 1)
            self.CL = int(2 * np.sum(AR * (W - 1)))
            self.crop = _Cropping1D((self.CL // 2, self.CL // 2))

        def forward(self, x):
            x = self.initial_conv(x)
            x, skip = self.initial_skip(x, 0)
            for m in self.residual_units:
                x, skip = m(x, skip)
            return F.softmax(self.final_conv(self.crop(skip)), dim=1)

    return SpliceAI_ReLU


# ----------------------------------------------------------------------------
# Input encoding — mirrors openspliceai's create_datapoints for positional parity
# ----------------------------------------------------------------------------

_IN_MAP = np.asarray(
    [[0, 0, 0, 0],
     [1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 1]],
    dtype=np.float32,
)


def _create_datapoints(seq: str, *, SL: int = 5000, CL_max: int = 10000) -> np.ndarray:
    allowed = set("ACGT")
    seq = "".join(c if c in allowed else "N" for c in seq.upper())
    seq = "N" * (CL_max // 2) + seq + "N" * (CL_max // 2)
    table = str.maketrans({"A": "1", "C": "2", "G": "3", "T": "4", "N": "0"})
    seq = seq.translate(table)

    arr = np.fromiter(map(int, seq), dtype=np.int8)
    win = SL + CL_max
    n_windows = int(np.ceil((len(arr) - CL_max) / SL))
    windows = np.zeros((n_windows, win), dtype=np.int8)
    for i in range(n_windows):
        chunk = arr[i * SL: i * SL + win]
        windows[i, : len(chunk)] = chunk
    return _IN_MAP[windows]  # (n_windows, win, 4)


# ----------------------------------------------------------------------------
# Adapter
# ----------------------------------------------------------------------------

class SpliceAIPyTorch(SplicingPredictor):
    """SpliceAI weights (Keras→PyTorch translation) in plain-ReLU PyTorch architecture."""

    name = "spliceai_pytorch"
    _CONTEXT = 5000  # CL_max=10000 → 5,000 each side

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
        d = _resolve("spliceai_pytorch")
        return Path(d) if d is not None else None

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
        if SpliceAIPyTorch._models is not None:
            return SpliceAIPyTorch._models, SpliceAIPyTorch._device

        import torch
        weights_dir = self._resolve_weights_dir()
        if weights_dir is None and self._dir_override is None:
            from ..weights import ensure_dir
            weights_dir = ensure_dir("spliceai_pytorch")  # auto-downloads on a miss
        if weights_dir is None:
            raise RuntimeError(
                "SpliceAI-PyTorch weights not found and could not be downloaded "
                "from the Hub. Check network / HF access, or run "
                "`oncosplice-download-weights spliceai_pytorch`."
            )

        SpliceAI_ReLU = _build_model_class()
        L, W, AR = _arch_10000()
        device = self._pick_device()

        weight_files = sorted(Path(weights_dir).glob("*.pt"))
        models = []
        for f in weight_files:
            m = SpliceAI_ReLU(L, W, AR).to(device)
            m.load_state_dict(torch.load(f, map_location=device))
            m.eval()
            models.append(m)
        SpliceAIPyTorch._models = models
        SpliceAIPyTorch._device = device
        SpliceAIPyTorch._weights_dir = Path(weights_dir)
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
            X = _create_datapoints(seq)                           # (nw, win, 4)
            X = torch.from_numpy(np.transpose(X, (0, 2, 1))).to(device)  # (nw, 4, win)

            with torch.no_grad():
                preds = torch.stack([m(X).detach().cpu() for m in models], dim=0)
                y = preds.mean(dim=0)                              # (nw, 3, SL)
            y = y.permute(0, 2, 1).contiguous().view(-1, y.shape[1]).numpy()  # (nw*SL, 3)
            y = y[:len(seq), :]                                    # crop to bio length
            y_bio = y[cl:-cl, :]                                   # strip context
            results.append(SplicingPrediction(acceptor=y_bio[:, 1], donor=y_bio[:, 2]))
        return results
