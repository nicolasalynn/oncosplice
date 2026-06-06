"""Splicing-engine adapter layer.

Every concrete predictor implements :class:`SplicingPredictor` — uniform
interface, uniform input/output shape. The factory :func:`get_predictor`
resolves user-friendly engine names to a concrete instance.

Production engines (recommended for inference):

- ``"openspliceai"`` (alias ``"spliceai"``) — OpenSpliceAI MANE PyTorch
  ensemble. **Default.**
- ``"spliceai_pytorch"`` (alias ``"spliceai_pt"`` / ``"spliceai_translated"``)
  — original Jaganathan-2019 SpliceAI weights translated to PyTorch (plain
  ReLU). Numerically identical to running the original Keras model
  (verified rho=1.000 across 100 test pairs); ~2.5× faster than the Keras
  path; GPU-friendly.
- ``"pangolin"`` — multi-tissue Pangolin ensemble.
- ``"spliceformer"`` — Spliceformer 40k-context transformer (experimental;
  see :mod:`oncosplice.engines.spliceformer`).
- ``"ensemble:a,b,c"`` — average the listed predictors. Aliases work too,
  e.g. ``"ensemble:spliceai,pangolin"``.
- ``"average"`` — shorthand for ``ensemble:openspliceai,pangolin``
  (matches the legacy oncosplice 3.0.0 behaviour).

Reference-only engine (use for one-time verification, not production):

- ``"spliceai_keras"`` (alias ``"spliceai_original"``) — original TF/Keras
  SpliceAI .h5 weights. Kept as the *reference implementation* for
  validating the PyTorch translation. **Prefer ``"spliceai_pytorch"`` for
  any actual work** — it produces identical numerical output without the
  TF/Keras dependency or the per-call retracing overhead.
"""
from __future__ import annotations

from typing import Callable, Dict

from .base import SplicingPrediction, SplicingPredictor
from .ensemble import EnsemblePredictor
from .openspliceai import OpenSpliceAI
from .pangolin import Pangolin
from .spliceai_keras import SpliceAIKeras
from .spliceai_pytorch import SpliceAIPyTorch
from .spliceformer import Spliceformer

__all__ = [
    "SplicingPredictor", "SplicingPrediction",
    "OpenSpliceAI", "SpliceAIKeras", "SpliceAIPyTorch",
    "Pangolin", "Spliceformer",
    "EnsemblePredictor",
    "get_predictor", "list_available_engines",
]


# Canonical name → factory
_REGISTRY: Dict[str, Callable[[], SplicingPredictor]] = {
    "openspliceai":      OpenSpliceAI,       # MANE-trained PyTorch (LeakyReLU class)
    "spliceai_keras":    SpliceAIKeras,      # original Illumina Keras .h5 (TF)
    "spliceai_pytorch":  SpliceAIPyTorch,    # Keras-translated weights in plain-ReLU PyTorch
    "pangolin":          Pangolin,
    "spliceformer":      Spliceformer,
}

# User-facing aliases (also case-insensitive). Resolved first to a canonical key.
_ALIASES: Dict[str, str] = {
    "spliceai":           "openspliceai",      # default — MANE PyTorch
    "spliceai_original":  "spliceai_keras",    # legacy name
    "spliceai_tf":        "spliceai_keras",
    "spliceai_pt":        "spliceai_pytorch",
    "spliceai_translated":"spliceai_pytorch",
    "sai":                "openspliceai",
    "open_spliceai":      "openspliceai",
    "osai":               "openspliceai",
    "pan":                "pangolin",
    "sf":                 "spliceformer",
}


def _canonical(name: str) -> str:
    key = name.strip().lower()
    return _ALIASES.get(key, key)


def get_predictor(name: str, **kwargs) -> SplicingPredictor:
    """Resolve a name to a :class:`SplicingPredictor` instance.

    Parameters
    ----------
    name
        Engine identifier. See module docstring for accepted values.
    **kwargs
        Forwarded to the predictor constructor (``device``, ``model_dir``,
        ``weights_dir`` …). Ignored for ensemble strings.
    """
    name = name.strip()

    # Ensemble syntax: ``ensemble:a,b,c`` or legacy ``average``.
    lname = name.lower()
    if lname == "average":
        return EnsemblePredictor(
            [OpenSpliceAI(**kwargs), Pangolin(**kwargs)],
            name="average",
        )
    if lname.startswith("ensemble:"):
        parts = [p.strip() for p in name.split(":", 1)[1].split(",") if p.strip()]
        if not parts:
            raise ValueError(f"empty ensemble spec: {name!r}")
        return EnsemblePredictor([get_predictor(p, **kwargs) for p in parts], name=name)

    key = _canonical(name)
    factory = _REGISTRY.get(key)
    if factory is None:
        valid = sorted(set(_REGISTRY) | set(_ALIASES) | {"average", "ensemble:<a,b,...>"})
        raise ValueError(
            f"Unknown splicing engine {name!r}. Valid names: {valid}"
        )
    return factory(**kwargs)


def list_available_engines() -> list[str]:
    """Return the canonical names of engines whose dependencies are importable
    on this machine. Useful for ``--help`` style diagnostics.
    """
    out: list[str] = []
    for key, factory in _REGISTRY.items():
        try:
            if factory().is_available():
                out.append(key)
        except Exception:
            continue
    return out
