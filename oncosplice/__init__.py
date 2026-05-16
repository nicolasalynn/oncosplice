"""
oncosplice — sequence-level splicing-epistasis pipeline.

Top-level layout:

- :class:`OncospliceEngine` — orchestrator. Requires ``geney`` + ``seqmat``.
- :mod:`oncosplice.engines` — standalone splice-site predictor adapters
  (``OpenSpliceAI``, ``SpliceAIPyTorch``, ``Pangolin``, ``Spliceformer``,
  ``EnsemblePredictor``). Importable without ``geney``. ``SpliceAIKeras`` is
  also available as the *reference implementation* for verification only —
  prefer ``SpliceAIPyTorch`` (identical output, ~2.5× faster, no TF dep).
- :mod:`oncosplice.scoring` — splicing / epistasis / Oncosplice scoring
  primitives. Pure-python; no model dependencies.
- :mod:`oncosplice.results` — typed dataclasses for results / protein library.
- :mod:`oncosplice.weights` — model-weight resolver + downloader CLI.

Quick start:

>>> from oncosplice import OncospliceEngine
>>> eng = OncospliceEngine(splicing_engine="openspliceai")
>>> pair = eng.analyze_pair("KRAS:12:25227343:G:T", "KRAS:12:25227344:A:T")
>>> print(pair.summary())

Predictor-only use (no geney / seqmat needed):

>>> from oncosplice.engines import get_predictor
>>> p = get_predictor("openspliceai")
>>> pred = p.predict(sequence)
"""
from __future__ import annotations

# Engines + weights are dependency-light and safe to import eagerly.
from .engines import (
    EnsemblePredictor,
    OpenSpliceAI,
    Pangolin,
    SpliceAIKeras,
    SpliceAIPyTorch,
    Spliceformer,
    SplicingPrediction,
    SplicingPredictor,
    get_predictor,
    list_available_engines,
)
from .results import (
    DoubleVariantResult,
    MissplicingProfile,
    MultiVariantResult,
    ProteinLibrary,
    SingleVariantResult,
    SiteEpistasis,
)
from .scoring.fingerprint import splicing_outcome_fingerprint, splicing_outcome_hash
from .variants import Variant, VariantPair

__version__ = "3.2.0"  # 3-bucket classifier: synergistic / rescue / compounding (+ non-epistatic fallback)

__all__ = [
    "OncospliceEngine",      # lazy attribute below
    "SingleVariantResult", "DoubleVariantResult", "MultiVariantResult",
    "SiteEpistasis", "MissplicingProfile", "ProteinLibrary",
    "splicing_outcome_fingerprint", "splicing_outcome_hash",
    "Variant", "VariantPair",
    "SplicingPredictor", "SplicingPrediction",
    "OpenSpliceAI", "SpliceAIKeras", "SpliceAIPyTorch", "Pangolin", "Spliceformer",
    "EnsemblePredictor",
    "get_predictor", "list_available_engines",
]


def __getattr__(name):
    """Lazy-load OncospliceEngine so that the rest of the package (engines,
    weights, scoring) is importable without geney/seqmat installed.
    """
    if name == "OncospliceEngine":
        from .engine import OncospliceEngine
        # Cache on the module so subsequent ``from oncosplice import X`` resolves
        # (PEP 562 __getattr__ + ``from … import …`` interacts oddly in some
        # CPython 3.13 builds; binding here is a robust workaround).
        globals()["OncospliceEngine"] = OncospliceEngine
        return OncospliceEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
